# Evaluation script for the model.
import torch
from envs import FinanceEnv, MultiFinanceEnv
from PPO import PPO
import numpy as np
from tqdm import tqdm
import random
from typing import Union

@torch.no_grad()
def testing_model(
    test_env: Union[FinanceEnv, MultiFinanceEnv],
    ppo_agent: PPO,
    episode_num: int,
):
    rewards = []
    epi_iter = tqdm(range(episode_num))
    for _ in epi_iter:
        state = test_env.reset()
        for t in range(test_env.episode_size):
            # Please ensure the model just be updated.
            # We need to ensure the old policy is equal to policy.
            action = ppo_agent.select_action(state)
            state, reward, done, _ = test_env.step(action)
            test_env.render()

            if done:
                epi_iter.set_description(f"Reward: {test_env.get_final_reward()}")
                rewards.append(test_env.get_final_reward())
                break
        ppo_agent.buffer.clear()

    return sum(rewards)/len(rewards)

def random_baseline(test_env: Union[MultiFinanceEnv, FinanceEnv], seed: int, episode_num: int):
    random.seed(seed)
    rewards = []
    epi_iter = tqdm(range(episode_num))
    for _ in epi_iter:
        state = test_env.reset()
        for t in range(test_env.episode_size):
            action = [random.uniform(-1, 1) for _ in range(test_env.action_dim)]
            state, reward, done, _ = test_env.step(action)
            # test_env.render()

            if done:
                epi_iter.set_description(f"Reward: {test_env.get_final_reward()}")
                rewards.append(test_env.get_final_reward())
                break
    return sum(rewards)/len(rewards)

if __name__ == "__main__":
    # Random seed
    seed = 42

    stock_list = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '2412.TW', '2308.TW', '2881.TW', '6505.TW', '2882.TW', '2303.TW', '1303.TW', '1301.TW', '2886.TW', '3711.TW', '2891.TW', '2002.TW', '1216.TW', '5880.TW', '2207.TW', '2884.TW']
    # Test environment setting.
    test_env_setting = {
        'start_date': '2021-01-01',
        'end_date': '2023-06-30',
        'tickers': stock_list,
        # 'state_feature_names': ['Open', 'Close', 'Adj Close'],
        'state_feature_names': ['Open', 'Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Close',
        'clip_action': True,
        'softmax_action': True,
        'use_final_reward': True,
        'extra_feature_names': ['moving_average', 'average', 'date', 'last_K_value', 'last_D_value', 'last_J_value', 'K_value', 'D_value', 'J_value'],
        'output_split_ticket_state': True,
        'log_return': True,
    }

    test_env = MultiFinanceEnv(**test_env_setting)

    # Random baseline
    print("Random baseline")
    print(random_baseline(test_env, seed, 20000))




