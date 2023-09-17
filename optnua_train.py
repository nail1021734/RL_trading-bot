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
import optuna
import copy

def train(trail: "optuna.Trail"):
    stock_list = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '2412.TW', '2308.TW', '2881.TW', '6505.TW', '2882.TW', '2303.TW', '1303.TW', '1301.TW', '2886.TW', '3711.TW', '2891.TW', '2002.TW', '1216.TW', '5880.TW', '2207.TW', '2884.TW']
    # stock_list = ['TSLA', 'AMD', 'NVDA', 'AAPL', 'TLRY']
    # Create environment config.
    # tanh_action = trail.suggest_categorical("tanh", [True, False])
    extra_feature_names = []
    if trail.suggest_categorical("moving_average", [True, False]):
        extra_feature_names.append('moving_average')
    if trail.suggest_categorical("average", [True, False]):
        extra_feature_names.append('average')
    if trail.suggest_categorical("date", [True, False]):
        extra_feature_names.append('date')
    if trail.suggest_categorical("last_K_value", [True, False]):
        extra_feature_names.append('K_value')
        extra_feature_names.append('D_value')
        extra_feature_names.append('J_value')
        extra_feature_names.append('last_K_value')
        extra_feature_names.append('last_D_value')
        extra_feature_names.append('last_J_value')
    if 'K_value' not in extra_feature_names and trail.suggest_categorical("K_value", [True, False]):
        extra_feature_names.append('K_value')
        extra_feature_names.append('D_value')
        extra_feature_names.append('J_value')

    tanh_action = False
    env_setting = {
        'start_date': '2010-01-01',
        'end_date': '2020-12-31',
        'tickers': stock_list,
        # 'state_feature_names': ['Open', 'Close', 'Adj Close'],
        'state_feature_names': ['Open', 'Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Close',
        'clip_action': tanh_action,
        'softmax_action': True,
        # 'use_final_reward': trail.suggest_categorical("use_final_reward", [True, False]),
        'use_final_reward': True,
        # 'extra_feature_names': ['moving_average', 'average', 'last_K_value', 'last_D_value', 'last_J_value', 'K_value', 'D_value', 'J_value'],
        'extra_feature_names': extra_feature_names,
        'output_split_ticket_state': True,
        # 'log_return': trail.suggest_categorical("log_return", [True, False]),
        'log_return': True,
    }

    # Create test environment.
    test_env_setting = copy.deepcopy(env_setting)
    test_env_setting['start_date'] = '2021-01-01'
    test_env_setting['end_date'] = '2021-06-30'

    # Initialize environments.
    env = MultiFinanceEnv(**env_setting)
    test_env = MultiFinanceEnv(**test_env_setting)

    # Initialize model config.
    model_config = ModelConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        ticker_num=len(stock_list),
        use_transformer_model=True,
        # actor_learning_rate=trail.suggest_float("actor_learning_rate", 5e-5, 1e-3),
        actor_learning_rate=0.0008855130541906845,
        # critic_learning_rate=trail.suggest_float("critic_learning_rate", 5e-5, 1e-3),
        critic_learning_rate=0.0006636017326283122,
        # gamma=trail.suggest_float("gamma", 0.7, 0.999),
        gamma=0.869325393673078,
        # update_iteration=trail.suggest_int("update_iteration", 1, 10),
        update_iteration=1,
        # clip=trail.suggest_float("clip", 0.1, 0.5),
        clip=0.12689399581918465,
        # vf_coef=trail.suggest_float("vf_coef", 0.2, 1.0),
        vf_coef=0.8306396541458023,
        # entropy_coef=trail.suggest_float("entropy_coef", 0.001, 0.05),
        entropy_coef=0.024118735627376767,
        has_continuous_action_space=True,
        tanh_action=tanh_action,
        # use_GAE=trail.suggest_categorical("use_GAE", [True, False]),
        use_GAE=False,
        fix_var_param=None,
        # fix_var_param=trail.suggest_categorical("fix_var_param", [None, {
            # 'init_std': 0.6,
            # 'decay_rate': 0.99,
            # 'min_value': 0.2,
            # 'decay_episode': 1000,
        # }])
        # fix_var_param={
            # 'init_std': trail.suggest_float("init_std", 0.4, 1.0),
            # 'decay_rate': trail.suggest_float("decay_rate", 0.7, 0.9999),
            # 'min_value': trail.suggest_float("min_value", 0.05, 0.3),
            # 'decay_episode': trail.suggest_int("decay_episode", 500, 2000),
        # }
    )

    # update_timestep = trail.suggest_int("update_timestep", 1, 50)
    update_timestep = 2
    # Initialize training config.
    config = Config(
        exp_name='test',
        model_config=model_config,
        # seed=42,
        seed=22,
        training_episode_num=20000,
        # Update model every 10 episode.
        # update_timestep=env.episode_size*10,
        update_timestep=env.episode_size*update_timestep,
        env_name='MultiFinanceEnv',
        env_kwargs=env_setting,
        test_env_kwargs=test_env_setting,
        # Log every 1000 episode.
        log_timestep=env.episode_size*update_timestep*10,
        # Save model every 10 update.
        save_timestep=env.episode_size*update_timestep*100,
    )

    # Save config.
    config.save_config()

    # Initialize summary writer.
    # writer = SummaryWriter(f'checkpoint/{config.exp_name}/log')

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
    test_reward = []
    all_train_rewards = []

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
                    all_train_rewards.append(env.get_final_reward())
                print(
                    f'Episode: {i_episode}, timestep: {timestep}, Last_10_rewards_avg: {sum(hst_final_rewards)/len(hst_final_rewards):.2f}')
                ppo_agent.update()

            if ppo_agent.fix_var is not None and i_episode % ppo_agent.fix_var.decay_episode == 0 and i_episode != 0:
                ppo_agent.fix_var.step()

            if timestep % config.log_timestep == 0 and timestep != 0:
                print('Logging...')
                # writer.add_scalar(
                    # 'rewards_avg',
                    # sum(hst_final_rewards)/len(hst_final_rewards),
                    # timestep
                # )
                hst_final_rewards = []

            if timestep % config.save_timestep == 0 and timestep != 0:
                # Evaluate model.
                print('Evaluating model...')
                mean_test_reward = testing_model(
                    test_env=test_env,
                    ppo_agent=ppo_agent,
                    episode_num=100,
                )
                test_reward.append(mean_test_reward)
                # writer.add_scalar(
                    # 'mean_test_rewards',
                    # mean_test_reward,
                    # timestep,
                # )
                # print('Saving model...')
                # ppo_agent.save(
                    # f'checkpoint/{config.exp_name}/ppo_agent_{timestep}.pth'
                # )


            if done:
                break
    return sum(test_reward)/len(test_reward), sum(all_train_rewards[-50:])/len(all_train_rewards[-50:])
    # return sum(test_reward)/len(test_reward)
    # Close writer.
    # writer.close()

if __name__ == '__main__':
    study = optuna.create_study(
        study_name='ppo_study_test_extra_feature2',
        storage='sqlite:///ppo_study2.db',
        load_if_exists=True,
        directions=['maximize', 'maximize'],
    )
    study.optimize(train, n_trials=100)

