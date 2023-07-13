import pytest
import numpy as np
from envs import FinanceEnv
import random
import pandas as pd
from datetime import datetime as dt
random.seed(42)


def test_FinanceEnv():
    def moving_average(
        episode_data: pd.DataFrame,
        day: int,
        state_dict: dict,
    ):
        r"""
        Calculate the moving average of the last 5 days.
        """
        if day == 0:
            return episode_data.iloc[day]['Close']
        else:
            return episode_data.iloc[max(day - 5, 0): day]['Close'].mean()

    env = FinanceEnv(
        start_date='2010-01-01',
        end_date='2020-12-31',
        ticker='AAPL',
        state_feature_names=['Open', 'High', 'Low', 'Close', 'Volume'],
        initial_balance=1000000,
        episode_size=30,
        initial_stock=0,
        target_feature='Close',
        extra_feature_dict={
            'Moving Average': moving_average,
        },
    )

    assert env.start_date == dt.strptime('2010-01-01', "%Y-%m-%d")
    assert env.end_date == dt.strptime('2020-12-31', "%Y-%m-%d")
    assert env.ticker == 'AAPL'
    assert env.episode_size == 30
    assert env.initial_stock == 0
    assert env.initial_balance == 1000000
    assert env.target_feature == 'Close'
    assert env.action_dim == 1
    assert env.state_dim == 8
    assert env.data.shape == (2768, 7)
    assert env.episode_data is None
    assert env.begin_date == 0
    assert env.day == 0
    assert env.balance == 1000000
    assert env.stock == 0

    # Test the reset function.
    state = env.reset()
    assert env.episode_data.shape == (31, 7)
    assert env.begin_date == 2619
    assert env.day == 0
    assert env.balance == 1000000
    assert env.stock == 0
    # Check the state.
    assert state.shape == (8, )
    assert state[0] == 1000000
    assert state[1] == 0
    assert state[2] == 79.4375
    assert state[3] == 80.5875015258789
    assert state[4] == 79.30249786376953
    assert state[5] == 80.4625015258789
    assert state[6] == 80791200.0
    assert state[7] == 80.4625015258789

    # Test the step function.
    # Test the case when the action is 1(buy all).
    next_state, reward, done, info = env.step(1)
    assert env.day == 1
    assert env.balance == 12.031036376953125
    assert env.stock == 12428
    # Check the next state.
    assert next_state.shape == (8, )
    assert next_state[0] == 12.031036376953125
    assert next_state[1] == 12428
    assert next_state[2] == 80.1875
    assert next_state[3] == 80.86000061035156
    assert next_state[4] == 79.73249816894531
    assert next_state[5] == 80.83499908447266
    assert next_state[6] == 87642800.0
    assert next_state[7] == 80.4625015258789
    # Check the reward.
    assert reward == 4629.399658203125
    # Check the done flag.
    assert done is False
    # Check the info.
    assert info == {}

    # Test get_final_reward function.
    final_reward = env.get_final_reward()
    assert final_reward == 4629.399658203125


