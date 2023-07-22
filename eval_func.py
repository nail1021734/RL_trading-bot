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
            action = [random.random() for _ in range(test_env.action_dim)]
            state, reward, done, _ = test_env.step(action)
            test_env.render()

            if done:
                epi_iter.set_description(f"Reward: {test_env.get_final_reward()}")
                rewards.append(test_env.get_final_reward())
                break
    return sum(rewards)/len(rewards)

if __name__ == "__main__":
    # Random seed
    seed = 42

    # Test environment setting
    test_env_setting = {
        'start_date': '2020-12-31',
        'end_date': None,
        'tickers': ['3231.TW', '2356.TW', '2610.TW', '2330.TW', '0050.TW'],
        'state_feature_names': ['Open', 'Close', 'Adj Close'],
        'initial_balance': 10000,
        'initial_stock': 0,
        'episode_size': 30,
        'target_feature': 'Adj Close',
        'extra_feature_dict': None,
    }
    test_env = MultiFinanceEnv(**test_env_setting)

    # Random baseline
    print("Random baseline")
    print(random_baseline(test_env, seed, 10000))




