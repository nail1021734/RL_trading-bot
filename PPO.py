import torch
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from typing import Union, Dict, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FixedVariance:
    def __init__(
        self,
        init_std: float,
        decay_rate: float,
        min_value: float,
        decay_episode: int,
    ):
        self.var = init_std**2
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.decay_episode = decay_episode

    def step(self):
        self.var = max(self.var * (self.decay_rate**2), self.min_value)

    def get_var(self):
        return self.var


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
        tanh_action: bool,
        has_continuous_action_space: bool,
        fix_var: Union[FixedVariance, None] = None,
    ):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.tanh_action = tanh_action
        self.action_dim = action_dim
        self.fix_var = fix_var

        if has_continuous_action_space:
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim*2) if fix_var is None else torch.nn.Linear(hidden_dim, action_dim),
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

    def forward(self):
        raise NotImplementedError

    def act(self, state: torch.Tensor):
        if self.has_continuous_action_space:
            actor_pred = self.actor(state)
            if self.fix_var is not None:
                action_mean = actor_pred
                if self.tanh_action:
                    action_mean = torch.nn.functional.tanh(action_mean)
                cov_mat = torch.diag_embed(torch.ones(self.action_dim) * self.fix_var.get_var()).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
            else:
                action_mean = actor_pred[:self.action_dim]
                if self.tanh_action:
                    action_mean = torch.nn.functional.tanh(action_mean)
                    action_var = torch.sigmoid(actor_pred[self.action_dim:]) + 1e-5
                else:
                    action_var = torch.nn.functional.softplus(actor_pred[self.action_dim:]) + 1e-5
                dist = MultivariateNormal(action_mean, torch.diag_embed(action_var))
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        if self.has_continuous_action_space:
            actor_pred = self.actor(state)
            if self.fix_var is not None:
                action_mean = actor_pred
                if self.tanh_action:
                    action_mean = torch.nn.functional.tanh(action_mean)

                # Set co-variance of distribution.
                action_var = torch.ones(self.action_dim) * self.fix_var.get_var()
                action_var = action_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(action_var).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
            else:
                action_mean = actor_pred[:, :self.action_dim]
                if self.tanh_action:
                    action_mean = torch.nn.functional.tanh(action_mean)
                    action_var = torch.sigmoid(actor_pred[:, self.action_dim:]) + 1e-5
                else:
                    action_var = torch.nn.functional.softplus(actor_pred[:, self.action_dim:]) + 1e-5
                dist = MultivariateNormal(action_mean, torch.diag_embed(action_var))
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
        vf_coef: float,
        entropy_coef: float,
        tanh_action: bool,
        use_GAE: bool,
        fix_var_param: Union[Dict, None],
    ):
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.clip = clip
        self.update_iteration = update_iteration
        self.buffer = RolloutBuffer()
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.tanh_action = tanh_action
        self.use_GAE = use_GAE
        if fix_var_param is None:
            self.fix_var = None
        else:
            self.fix_var = FixedVariance(
                init_std=fix_var_param['init_std'],
                decay_rate=fix_var_param['decay_rate'],
                min_value=fix_var_param['min_value'],
                decay_episode=fix_var_param['decay_episode'],
            )

        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            tanh_action=tanh_action,
            has_continuous_action_space=has_continuous_action_space,
            fix_var=self.fix_var,
        ).to(device)

        opt_list = [
            {'params': self.policy.actor.parameters(), 'lr': actor_learning_rate},
            {'params': self.policy.critic.parameters(), 'lr': critic_learning_rate},
        ]
        self.optimizer = torch.optim.Adam(opt_list)

        self.old_policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            tanh_action=tanh_action,
            has_continuous_action_space=has_continuous_action_space,
            fix_var=self.fix_var,
        ).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.mse_loss = torch.nn.MSELoss()

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
            if self.use_GAE:
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            else:
                rewards.insert(0, reward)
        # print(sum(self.buffer.rewards))

        # Normalizing the rewards.
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # Convert list to tensor.
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.2)
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
