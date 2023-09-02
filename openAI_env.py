import gym
import torch
import sys
from envs import FinanceEnv, MultiFinanceEnv
from config import ModelConfig, Config
from PPO import PPO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from eval_func import testing_model
from utils import average, mov_average
import gc


if __name__ == '__main__':
    # Initialize environments.
    env = gym.make('LunarLanderContinuous-v2')

    # Initialize model config.
    model_config = ModelConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=256,
        actor_learning_rate=5e-4,
        critic_learning_rate=3e-4,
        gamma=0.99,
        update_iteration=10,
        clip=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        has_continuous_action_space=True,
        tanh_action=True,
        use_GAE=True,
    )

    # Initialize training config.
    config = Config(
        exp_name='LunarLanderContinuous_tanh_exp1',
        model_config=model_config,
        # seed=42,
        seed=22,
        training_timestep=500000000,
        # Update model every 10 episode.
        update_timestep=64,
        env_name='MultiFinanceEnv',
        env_kwargs={},
        test_env_kwargs={},
        # Log every 1000 episode.
        log_timestep=10*64,
        # Save model every 10 update.
        save_timestep=1000*1000,
    )

    # Save config.
    config.save_config()

    # Initialize summary writer.
    writer = SummaryWriter(f'checkpoint/{config.exp_name}/log')

    # Set random seed.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Initialize PPO agent.
    ppo_agent = PPO(
        state_dim=model_config.state_dim,
        action_dim=model_config.action_dim,
        hidden_dim=model_config.hidden_dim,
        actor_learning_rate=model_config.actor_learning_rate,
        critic_learning_rate=model_config.critic_learning_rate,
        gamma=model_config.gamma,
        update_iteration=model_config.update_iteration,
        clip=model_config.clip,
        has_continuous_action_space=model_config.has_continuous_action_space,
        vf_coef=model_config.vf_coef,
        entropy_coef=model_config.entropy_coef,
        tanh_action=model_config.tanh_action,
        use_GAE=model_config.use_GAE,
    )

    # Training loop.
    timestep = 0
    # History rewards
    hst_final_rewards = []
    while timestep <= config.training_timestep:
        state, _ = env.reset()
        epi_reward = 0
        for t in range(1, 1000):
            timestep += 1
            # Select action.
            action = ppo_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            epi_reward += reward

            # Append reward and is_terminal to rollout buffer.
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            # Update PPO agent.
            state = next_state

            # Update PPO agent.
            if timestep % config.update_timestep == 0 and timestep != 0:
                print(epi_reward)
                ppo_agent.update()

            if timestep % config.log_timestep == 0 and timestep != 0:
                print('Logging...')
                writer.add_scalar(
                    'rewards_avg',
                    epi_reward,
                    timestep
                )
                hst_final_rewards = []

            if done:
                break
        gc.collect()

    # Close writer.
    writer.close()
    # Close env.
    env.close()

