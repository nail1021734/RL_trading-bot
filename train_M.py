import torch
import sys
from envs import FinanceEnv, MultiFinanceEnv
from config import ModelConfig, Config
from PPO import PPO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from eval_func import testing_model
import gc



if __name__ == '__main__':
    # TODO:
    # 1. Select and add more ticker.
    # 2. Test GAE.
    # 3. Add more feature.

    stock_list = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '2412.TW', '2308.TW', '2881.TW', '6505.TW', '2882.TW', '2303.TW', '1303.TW', '1301.TW', '2886.TW', '3711.TW', '2891.TW', '2002.TW', '1216.TW', '5880.TW', '2207.TW', '2884.TW']
    # Create environment config.
    env_setting = {
        'start_date': '2010-01-01',
        'end_date': '2020-12-31',
        'tickers': stock_list,
        'state_feature_names': ['Open', 'Close', 'Adj Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Adj Close',
        'clip_action': True,
        'softmax_action': False,
        'extra_feature_names': ['moving_average', 'average', 'date', 'last_K_value', 'last_D_value', 'last_J_value', 'K_value', 'D_value', 'J_value'],
        'output_split_ticket_state': True,
    }

    # Create test environment.
    test_env_setting = {
        'start_date': '2020-12-31',
        'end_date': None,
        'tickers': stock_list,
        'state_feature_names': ['Open', 'Close', 'Adj Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Adj Close',
        'clip_action': True,
        'softmax_action': False,
        'extra_feature_names': ['moving_average', 'average', 'date', 'last_K_value', 'last_D_value', 'last_J_value', 'K_value', 'D_value', 'J_value'],
        'output_split_ticket_state': True,
    }

    # Initialize environments.
    env = MultiFinanceEnv(**env_setting)
    test_env = MultiFinanceEnv(**test_env_setting)

    # Initialize model config.
    model_config = ModelConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
        ticker_num=len(stock_list),
        use_transformer_model=True,
        actor_learning_rate=5e-4,
        critic_learning_rate=3e-4,
        gamma=0.925,
        update_iteration=10,
        clip=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        has_continuous_action_space=True,
        tanh_action=True,
        use_GAE=True,
        fix_var_param=None,
        # fix_var_param={
            # 'init_std': 0.6,
            # 'decay_rate': 0.9999,
            # 'min_value': 0.2,
            # 'decay_episode': 1000,
        # }
    )

    # Initialize training config.
    config = Config(
        exp_name='test',
        model_config=model_config,
        # seed=42,
        seed=22,
        training_episode_num=50000,
        # Update model every 10 episode.
        update_timestep=env.episode_size*10,
        env_name='MultiFinanceEnv',
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
        ticker_num=model_config.ticker_num,
        actor_learning_rate=model_config.actor_learning_rate,
        critic_learning_rate=model_config.critic_learning_rate,
        gamma=model_config.gamma,
        update_iteration=model_config.update_iteration,
        clip=model_config.clip,
        has_continuous_action_space=model_config.has_continuous_action_space,
        vf_coef=model_config.vf_coef,
        entropy_coef=model_config.entropy_coef,
        tanh_action=model_config.tanh_action,
        use_transformer_model=model_config.use_transformer_model,
        use_GAE=model_config.use_GAE,
        fix_var_param=model_config.fix_var_param,
    )

    # Training loop.
    timestep = 0
    # History rewards
    hst_final_rewards = []
    for i_episode in range(config.training_episode_num):
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
                if done:
                    hst_final_rewards.append(env.get_final_reward())
                print(
                    f'Episode: {i_episode}, timestep: {timestep}, Last_10_rewards_avg: {sum(hst_final_rewards)/len(hst_final_rewards):.2f}')
                ppo_agent.update()

            if ppo_agent.fix_var is not None and i_episode % ppo_agent.fix_var.decay_episode == 0 and i_episode != 0:
                ppo_agent.fix_var.step()

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


            if done:
                break

    # Close writer.
    writer.close()

