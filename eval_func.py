# Evaluation script for the model.
import torch
from envs import FinanceEnv
from PPO import PPO
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def testing_model(
    test_env: FinanceEnv,
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
