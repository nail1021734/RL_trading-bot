import torch
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        has_continuous_action_space: bool,
        action_std_init: float,
        trainable_std = False,
    ):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.trainable_std = trainable_std

        self.action_dim = action_dim
        if has_continuous_action_space and trainable_std == False:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        if has_continuous_action_space:
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim),
                torch.nn.Tanh(),
            )
        else:
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim),
                torch.nn.Softmax(dim=-1),
            )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

        if has_continuous_action_space and trainable_std == True:
            self.trainable_cov_mat = torch.nn.Parameter(torch.ones(action_dim) * (action_std_init * action_std_init))
            self.action_var = self.trainable_cov_mat.to(device)
            # Why need retain_grad()?
            self.action_var.retain_grad()

    def set_action_std(self, new_action_std: float):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,),
                new_action_std * new_action_std
            ).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state: torch.Tensor):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            if not self.trainable_std:
                cov_mat = torch.diag_embed(self.action_var).unsqueeze(dim=0)
            else:
                cov_mat = torch.diag_embed(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            if not self.trainable_std:
                cov_mat = torch.diag_embed(action_var)
            else:
                cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        update_iteration: int,
        clip: float,
        has_continuous_action_space: bool,
        action_std_init: float,
        vf_coef: float,
        entropy_coef: float,
        trainable_std: bool = False,
    ):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.clip = clip
        self.update_iteration = update_iteration
        self.buffer = RolloutBuffer()
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.trainable_std = trainable_std

        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            trainable_std=trainable_std,
        ).to(device)

        opt_list = [
            {'params': self.policy.actor.parameters(), 'lr': actor_learning_rate},
            {'params': self.policy.critic.parameters(), 'lr': critic_learning_rate},
        ]
        if has_continuous_action_space and trainable_std:
            opt_list.append(
                {'params': self.policy.action_var, 'lr': actor_learning_rate}
            )
        self.optimizer = torch.optim.Adam(opt_list)

        self.old_policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            trainable_std=trainable_std,
        ).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.mse_loss = torch.nn.MSELoss()

    def set_action_std(self, new_action_std: float):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            self.action_std = max(self.action_std, min_action_std)
            self.set_action_std(self.action_std)

    def select_action(self, state: np.ndarray):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.old_policy.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.is_terminals),
        ):
            # if is_terminal:
                # discounted_reward = 0
            # discounted_reward = reward + (self.gamma * discounted_reward)
            # rewards.insert(0, discounted_reward)
            rewards.insert(0, reward)
        # print(sum(self.buffer.rewards))

        # Normalizing the rewards.
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor.
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # Train policy with multiple epochs.
        for _ in range(self.update_iteration):
            # Evaluating old actions and values.
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()

            # Finding the ratio (pi_theta / pi_theta__old).
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss.
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            # Final Loss of clipped objective PPO.
            loss = -torch.min(surr1, surr2) + self.vf_coef * self.mse_loss(state_values, rewards) - self.entropy_coef * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # print(self.policy.action_var)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Copy new weights into old policy.
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Clear buffer.
        self.buffer.clear()

    def save(self, checkpoint_path: str):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str):
        self.policy.load_state_dict(torch.load(checkpoint_path))
        self.old_policy.load_state_dict(torch.load(checkpoint_path))
