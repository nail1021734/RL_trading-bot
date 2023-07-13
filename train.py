import torch
from envs import FinanceEnv
from config import ModelConfig, Config
from PPO import PPO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from eval_func import testing_model


if __name__ == '__main__':
    # Create environment config.
    env_setting = {
        'start_date': '2010-01-01',
        'end_date': '2020-12-31',
        'ticker': '2330.TW',
        'state_feature_names': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Adj Close',
        'extra_feature_dict': None,
    }

    # Create test environment.
    test_env_setting = {
        'start_date': '2020-12-31',
        'end_date': None,
        'ticker': '2330.TW',
        'state_feature_names': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Adj Close',
        'extra_feature_dict': None,
    }

    # Initialize environments.
    env = FinanceEnv(**env_setting)
    test_env = FinanceEnv(**test_env_setting)

    # Initialize model config.
    model_config = ModelConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
        actor_learning_rate=5e-4,
        critic_learning_rate=3e-4,
        gamma=0.99,
        update_iteration=10,
        action_std_init=0.6,
        clip=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        has_continuous_action_space=True,
    )

    # Initialize training config.
    config = Config(
        exp_name='exp1',
        model_config=model_config,
        seed=42,
        training_timestep=20000000,
        # Update model every 10 episode.
        update_timestep=env.episode_size*10,
        action_std_decay_timestep=env.episode_size*1000,
        min_action_std=0.2,
        env_name='FinanceEnv',
        env_kwargs=env_setting,
        test_env_kwargs=test_env_setting,
        # Log every 1000 episode.
        log_timestep=env.episode_size*100,
        # Save model every 10 update.
        save_timestep=env.episode_size*1000,
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
        action_std_init=model_config.action_std_init,
        vf_coef=model_config.vf_coef,
        entropy_coef=model_config.entropy_coef,
    )

    # Training loop.
    timestep = 0
    # History rewards
    hst_final_rewards = []
    for i_episode in range(config.training_timestep):
        state = env.reset()
        for _ in range(env.episode_size):
            timestep += 1
            # Select action.
            action = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Append reward and is_terminal to rollout buffer.
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            # Update PPO agent.
            state = next_state

            # Update PPO agent.
            if timestep % config.update_timestep == 0 and timestep != 0:
                print(
                    f'Episode: {i_episode}, timestep: {timestep}, Last_10_rewards_avg: {sum(hst_final_rewards[-10:])/min(len(hst_final_rewards), 10):.2f}')
                ppo_agent.update()

            if timestep % config.log_timestep == 0 and timestep != 0:
                print('Logging...')
                writer.add_scalar(
                    'rewards_avg',
                    sum(hst_final_rewards)/len(hst_final_rewards),
                    timestep
                )
                hst_final_rewards = []

            if timestep % config.save_timestep == 0 and timestep != 0:
                # Evaluate model.
                print('Evaluating model...')
                mean_test_reward = testing_model(
                    test_env=test_env,
                    ppo_agent=ppo_agent,
                    episode_num=100,
                )
                writer.add_scalar(
                    'mean_test_rewards',
                    mean_test_reward,
                    timestep,
                )
                print('Saving model...')
                ppo_agent.save(
                    f'checkpoint/{config.exp_name}/ppo_agent_{timestep}.pth'
                )

            if model_config.has_continuous_action_space and timestep % config.action_std_decay_timestep == 0 and timestep != 0:
                ppo_agent.decay_action_std(
                    action_std_decay_rate=config.action_std_decay_timestep,
                    min_action_std=config.min_action_std,
                )

            if done:
                hst_final_rewards.append(env.get_final_reward())
                break

    # Close writer.
    writer.close()

