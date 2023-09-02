# Create a environment to simulate real world stock trading.
from pandas_datareader import data as pdr
import random
import gc
import numpy as np
import os, psutil
from datetime import datetime as dt
from typing import Union, List, Dict, Tuple, Optional, Callable
import math
import yfinance as yf
import torch
yf.pdr_override()


class FinanceEnv:
    r"""
    Create a environment to simulate real world stock trading.
    Step function used to return the next state and reward after doing some
    action.
    """
    def __init__(
        self,
        start_date: str,
        end_date: str = None,
        ticker: str = 'AAPL',
        state_feature_names: List = ['Adj Close'],
        initial_balance: int = 10000,
        episode_size: int = 30,
        initial_stock: int = 0,
        target_feature: str = 'Adj Close',
        extra_feature_dict: Dict[str, Callable] = None,
    ):
        r"""
        Initialize the environment.
        """
        super().__init__()
        # Save the parameters.
        self.start_date = dt.strptime(start_date, "%Y-%m-%d")
        # If end_date is None, set it to today.
        if end_date is None:
            self.end_date = dt.now()
        else:
            self.end_date = dt.strptime(end_date, "%Y-%m-%d")

        self.ticker = ticker
        self.episode_size = episode_size
        self.initial_stock = initial_stock
        self.initial_balance = initial_balance
        self.target_feature = target_feature
        self.action_dim = 1
        self.state_feature_names = state_feature_names
        self.extra_feature_dict = extra_feature_dict
        # Features, now balance and now stock.
        self.state_dim = 2 + len(state_feature_names)
        if extra_feature_dict is not None:
            self.state_dim += len(extra_feature_dict)

        # Get the data.
        self.data = pdr.get_data_yahoo(
            self.ticker, start=self.start_date, end=self.end_date)
        self.data = self.data.dropna()
        self.data = self.data.reset_index()

        # Initialize the state features.
        self.episode_data = None
        self.begin_date = 0
        self.day = 0
        self.balance = initial_balance
        self.stock = initial_stock

    def get_state(self):
        r"""
        Get the state of the environment.
        """
        # Initialize the state.
        state = [
            self.balance,
            self.stock,
        ]

        # Initialize the state dict.
        state_dict = {
            'balance': self.balance,
            'stock': self.stock,
        }

        # Add the state features.
        for name in self.state_feature_names:
            state_dict[name] = self.episode_data.iloc[self.day][name]
            state.append(self.episode_data.iloc[self.day][name])

        # Add the extra features.
        if self.extra_feature_dict is not None:
            for name, func in self.extra_feature_dict.items():
                state.append(
                    func(
                        episode_data=self.episode_data,
                        day=self.day,
                        state_dict=state_dict,
                    )
                )

        return np.array(state)

    def reset(self, rand: bool = True):
        r"""
        Reset the environment.
        """
        # Reset features.
        self.day = 0
        self.balance = self.initial_balance
        self.stock = self.initial_stock

        # Sample a date interval randomly.
        if rand:
            self.begin_date = random.randint(
                0,
                len(self.data) - self.episode_size - 1,
            )
        else:
            self.begin_date = 0
        # Get the episode data.
        self.episode_data = self.data.iloc[self.begin_date: self.begin_date + self.episode_size+1].reset_index(drop=True)

        return self.get_state()

    def step(
        self,
        action: float,
    ):
        r"""
        Return the next state and reward after doing some action.
        """
        # Convert the action to a float between -1 and 1. (tanh)
        # action = (math.exp(action) - math.exp(-action)) / (math.exp(action) + math.exp(-action))
        if action > 1:
            action = 1
        elif action < -1:
            action = -1

        # Calculate the action bound.
        action_bound = {
            'min': -self.stock,
            'max': self.balance//self.episode_data.iloc[self.day][self.target_feature]
        }

        action = action_bound['min'] + (action_bound['max'] - action_bound['min'])/2*(action + 1)

        action = int(action)

        # Update balance and stock.
        self.balance -= self.episode_data.iloc[self.day][self.target_feature] * action
        self.stock += action

        # Check if the episode is done.
        done = self.day == (self.episode_size - 1)

        # Update the day.
        self.day = min(1 + self.day, self.episode_size)

        # Calculate the reward.
        current_portfolio = self.balance + self.stock * self.episode_data.iloc[self.day-1][self.target_feature]
        next_portfolio = self.balance + self.stock * self.episode_data.iloc[self.day][self.target_feature]
        reward = next_portfolio - current_portfolio
        # print(reward)

        # Return the next state and reward.
        next_state = self.get_state()

        return next_state, reward, done, {}

    def render(self):
        r"""
        Render the environment.
        """
        print(
            "Day: {} | Balance: {} | Stock: {} | Price: {}".format(
                self.day,
                self.balance,
                self.stock,
                self.episode_data.iloc[self.day][self.target_feature],
            )
        )

    def close(self):
        r"""
        Close the environment.
        """
        pass

    def get_final_reward(self):
        return self.balance + self.stock * self.episode_data.iloc[self.day][self.target_feature] - self.initial_balance


class MultiFinanceEnv:
    r"""
    Create a environment to simulate real world stock trading.
    Step function used to return the next state and reward after doing some
    action.
    """
    def __init__(
        self,
        start_date: str,
        end_date: str = None,
        tickers: List = ['AAPL', '2330.TW'],
        state_feature_names: List = ['Adj Close'],
        initial_balance: int = 10000,
        episode_size: int = 30,
        initial_stock: int = 0,
        softmax_action: bool = False,
        target_feature: str = 'Adj Close',
        clip_action: bool = False,
        extra_feature_dict: Dict[str, Callable] = None,
        use_final_reward: bool = False,
    ):
        r"""
        Initialize the environment.
        """
        super().__init__()
        # Save the parameters.
        self.start_date = dt.strptime(start_date, "%Y-%m-%d")
        # If end_date is None, set it to today.
        if end_date is None:
            self.end_date = dt.now()
        else:
            self.end_date = dt.strptime(end_date, "%Y-%m-%d")

        self.softmax_action = softmax_action
        self.clip_action = clip_action
        self.tickers = tickers
        self.episode_size = episode_size
        self.initial_stock = initial_stock
        self.initial_balance = initial_balance
        self.target_feature = target_feature
        self.action_dim = len(tickers)
        self.state_feature_names = state_feature_names
        self.extra_feature_dict = extra_feature_dict
        # Features, now balance and now stock.
        self.state_dim = 1 + (len(state_feature_names) + 1)*len(tickers) + 1
        if extra_feature_dict is not None:
            self.state_dim += (len(extra_feature_dict)-1)*len(tickers)
        self.use_final_reward = use_final_reward

        # Get the data.
        self.data = pdr.get_data_yahoo(
            self.tickers, start=self.start_date, end=self.end_date)
        self.data = self.data.dropna()
        self.data_date = self.data.index.to_list()
        self.data = self.data.reset_index(drop=True).to_dict('list')
        tmp = {t: {} for t in self.tickers}
        for key, val in self.data.items():
            feature, ticker = key
            tmp[ticker][feature] = val
        self.data = tmp

        # Initialize the state features.
        self.episode_data = None
        self.begin_date = 0
        self.day = 0
        self.balance = initial_balance
        self.stock = {t: initial_stock for t in tickers}

    def get_state(self):
        r"""
        Get the state of the environment.
        """
        # Initialize the state.
        state = [self.balance] + [self.stock[t] for t in self.tickers]

        # Initialize the state dict.
        state_dict = {
            'balance': self.balance,
            'stock': self.stock,
        }

        # Add the state features.
        for t in self.tickers:
            for name in self.state_feature_names:
                state_dict[t+'_'+name] = self.episode_data[t][name][self.day]
                state.append(self.episode_data[t][name][self.day])

        # Add the extra features.
        if self.extra_feature_dict is not None:
            for name, func in self.extra_feature_dict.items():
                func(
                    tickers=self.tickers,
                    episode_data=self.episode_data,
                    day=self.day,
                    state_dict=state_dict,
                    state=state,
                    target_feature=self.target_feature,
                    begin_date=self.begin_date,
                    data_date=self.data_date,
                )

        return np.array(state)

    def reset(self, rand: bool = True):
        r"""
        Reset the environment.
        """
        # Reset features.
        self.day = 0
        self.balance = self.initial_balance
        self.stock = {t: self.initial_stock for t in self.tickers}

        # Sample a date interval randomly.
        if rand:
            self.begin_date = random.randint(
                0,
                len(self.data_date) - self.episode_size - 1,
            )
        else:
            self.begin_date = 0
        # Get the episode data.
        self.episode_date = self.data_date[self.begin_date: self.begin_date + self.episode_size+1]
        self.episode_data = {t: {} for t in self.tickers}
        for t, val in self.data.items():
            for key, value in val.items():
                self.episode_data[t][key] = value[self.begin_date: self.begin_date + self.episode_size+1]
        # self.episode_data = self.data.iloc[self.begin_date: self.begin_date + self.episode_size+1].reset_index(drop=True)

        return self.get_state()

    def step(
        self,
        action: Union[float, List[float]],
    ):
        r"""
        Return the next state and reward after doing some action.
        """
        # Initialize the total reward.
        total_reward = 0

        # Check if the episode is done.
        done = self.day == (self.episode_size - 1)

        # Update the day.
        self.day = min(1 + self.day, self.episode_size)

        # Sell stock.
        for a, t in zip(action, self.tickers):
            if a >= 0:
                continue
            # Clip the action to a float between -1 and 1.
            if self.clip_action:
                if a < -1:
                    a = -1
                min_value = -self.stock[t]
                max_value = 0.0
                a = min_value + (max_value - min_value) * (a + 1) / 2

            # Clip the action to the action bound.
            if a < -self.stock[t]:
                a = -self.stock[t]

            a = int(a)

            # Update balance and stock.
            self.balance -= self.episode_data[t][self.target_feature][self.day-1] * a
            self.stock[t] += a

            # Calculate the reward.
            current_portfolio = self.balance + self.stock[t] * self.episode_data[t][self.target_feature][self.day-1]
            next_portfolio = self.balance + self.stock[t] * self.episode_data[t][self.target_feature][self.day]
            total_reward += next_portfolio - current_portfolio

        # Use softmax to calculate weight of each stock.
        if self.softmax_action:
            weight_action = torch.tensor(action)
            weight_action /= weight_action.max()
            weight_action[weight_action <= 0] = float('-inf')
            weight_action = torch.nn.functional.softmax(weight_action, dim=-1)
            weight_balance = weight_action * self.balance
            weight_balance = weight_balance.tolist()
            weighted_balance = {t: b for b, t in zip(weight_balance, self.tickers)}


        # Buy stock.
        for a, t in zip(action, self.tickers):
            if a < 0:
                continue
            # Convert the action to a float between -1 and 1.
            if self.clip_action:
                if a > 1:
                    a = 1
                min_value = 0.0
                if self.softmax_action:
                    max_value = weighted_balance[t]//self.episode_data[t][self.target_feature][self.day-1]
                else:
                    max_value = self.balance//self.episode_data[t][self.target_feature][self.day-1]
                a = min_value + (max_value - min_value) * (a + 1) / 2

            # Calculate the action bound.
            if self.softmax_action:
                bound = weighted_balance[t]//self.episode_data[t][self.target_feature][self.day-1]
                if a > bound:
                    a = bound
            else:
                bound = self.balance//self.episode_data[t][self.target_feature][self.day-1]
                if a > bound:
                    a = bound

            a = int(a)

            # Update balance and stock.
            self.balance -= self.episode_data[t][self.target_feature][self.day-1] * a
            self.stock[t] += a

            # Calculate the reward.
            current_portfolio = self.balance + self.stock[t] * self.episode_data[t][self.target_feature][self.day-1]
            next_portfolio = self.balance + self.stock[t] * self.episode_data[t][self.target_feature][self.day]
            total_reward += next_portfolio - current_portfolio
            # print(reward)
        # Return the next state and reward.
        next_state = self.get_state()

        if done and self.use_final_reward:
            total_reward = self.get_final_reward()

        return next_state, total_reward, done, {}

    def render(self):
        r"""
        Render the environment.
        """
        for t in self.tickers:
            print(
                "Day: {:3} |balance {:8}| {:8} |stock: {:8}|price: {:8}".format(
                    self.day,
                    self.balance,
                    t,
                    self.stock[t],
                    self.episode_data[t][self.target_feature][self.day],
                )
            )

    def close(self):
        r"""
        Close the environment.
        """
        pass

    def get_final_reward(self):
        r"""
        Get the final reward.
        """
        now_balance = self.balance
        for t in self.tickers:
            now_balance += self.stock[t] * self.episode_data[t][self.target_feature][self.day]

        return now_balance - self.initial_balance

