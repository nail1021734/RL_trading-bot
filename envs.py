# Create a environment to simulate real world stock trading
from utils import ExtraFeatureProcessor
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
        extra_feature_names: List[str] = None,
        use_final_reward: bool = False,
        output_split_ticket_state: bool = False,
        log_return: bool = False,
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

        self.tickers = tickers
        self.state_feature_names = state_feature_names
        self.initial_balance = initial_balance
        self.episode_size = episode_size
        self.initial_stock = initial_stock
        self.softmax_action = softmax_action
        self.target_feature = target_feature
        self.clip_action = clip_action
        self.extra_feature_names = extra_feature_names
        self.extra_feature_processor = ExtraFeatureProcessor(
            extra_feature_names=extra_feature_names,
        )
        self.use_final_reward = use_final_reward
        self.output_split_ticket_state = output_split_ticket_state
        self.log_return = log_return

        # Calculate the state dimension and action dimension.
        self.action_dim = len(tickers)
        if self.output_split_ticket_state:
            self.state_dim = 2 + len(state_feature_names)
            general_feature_num, feature_num = \
                self.extra_feature_processor.get_extra_feature_num(
                    env=self,
                )
            self.state_dim += general_feature_num + feature_num
        else:
            self.state_dim = 1 + (len(state_feature_names)+1)*len(tickers)
            general_feature_num, feature_num = \
                self.extra_feature_processor.get_extra_feature_num(
                    env=self,
                )
            self.state_dim += general_feature_num + feature_num*len(tickers)

        # Get the data.
        self.data = pdr.get_data_yahoo(
            self.tickers, start=self.start_date, end=self.end_date)
        self.data = self.data.dropna()
        self.date_index = self.data.index.to_list()
        self.data = self.data.reset_index(drop=True).to_dict('list')
        tmp = {t: {} for t in self.tickers}
        for key, val in self.data.items():
            feature, ticker = key
            tmp[ticker][feature] = val
        self.data = tmp

        # Initialize the state features.
        # episode_data format: {ticker: {feature: [day1_val, day2_val, ...], feature2: [...], ...}}
        self.episode_data = None
        self.begin_date = 0
        self.day = 0
        self.balance = initial_balance
        self.stock = {t: initial_stock for t in tickers}

    def get_state(self):
        r"""
        Get the state of the environment.
        """
        # Initialize the state dict.
        state_dict = {t: {} for t in self.tickers}

        # Add feature in state dict.
        state_dict['balance'] = self.balance
        for t in self.tickers:
            state_dict[t]['stock'] = self.stock[t]
        for t in self.tickers:
            for name in self.state_feature_names:
                state_dict[t][name] = self.episode_data[t][name][self.day]

        # Add the extra features.
        self.extra_feature_processor.process(
            state_dict=state_dict,
            env=self,
        )

        # Process the output features which is depend on `self.output_split_ticket_state`.
        state = []
        if self.output_split_ticket_state:
            for t in self.tickers:
                state.append([])
                state[-1].append(state_dict['balance'])
                state[-1].append(state_dict[t]['stock'])
                for feature_name in self.state_feature_names:
                    state[-1].append(state_dict[t][feature_name])
                for feature_name in self.extra_feature_names:
                    if feature_name in self.extra_feature_processor.general_functions.keys():
                        state[-1].append(state_dict[feature_name])
                    else:
                        state[-1].append(state_dict[t][feature_name])
        else:
            state.append(state_dict['balance'])
            for feature_name in self.extra_feature_names:
                if feature_name in self.extra_feature_processor.general_functions.keys():
                    state.append(state_dict[feature_name])
            for t in self.tickers:
                state.append(state_dict[t]['stock'])
                for feature_name in self.state_feature_names:
                    state.append(state_dict[t][feature_name])
                for feature_name in self.extra_feature_names:
                    if feature_name in self.extra_feature_processor.functions.keys():
                        state.append(state_dict[t][feature_name])

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
                len(self.date_index) - self.episode_size - 1,
            )
        else:
            self.begin_date = 0

        # Get the episode data.
        self.episode_data = {t: {} for t in self.tickers}
        for t, val in self.data.items():
            for key, value in val.items():
                self.episode_data[t][key] = value[self.begin_date: self.begin_date + self.episode_size+1]

        return self.get_state()

    def step(
        self,
        action: Union[float, List[float]],
    ):
        r"""
        Return the next state and reward after doing some action.
        """
        # Initialize the total reward.
        total_portfolio = 0

        # Check if the episode is done.
        done = self.day == (self.episode_size - 1)

        # Update the day.
        self.day = min(1 + self.day, self.episode_size)

        # Temporary store the current portfolio.
        tmp_portfolio = self.balance + sum([self.stock[t] * self.episode_data[t][self.target_feature][self.day-1] for t in self.tickers])

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
                # Convert [-1, 0] to [min_value, max_value].
                a = min_value + (max_value - min_value) * (a + 1)

            # Clip the action to the action bound.
            if a < -self.stock[t]:
                a = -self.stock[t]

            a = int(a)

            # Update balance and stock.
            self.balance -= self.episode_data[t][self.target_feature][self.day-1] * a
            self.stock[t] += a

            # Calculate the reward.
            next_portfolio = self.stock[t] * self.episode_data[t][self.target_feature][self.day]
            total_portfolio += next_portfolio

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
                # Convert [0, 1] to [min_value, max_value].
                a = min_value + (max_value - min_value) * a

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
            next_portfolio = self.stock[t] * self.episode_data[t][self.target_feature][self.day]
            total_portfolio += next_portfolio

        # Return the next state and reward.
        next_state = self.get_state()

        # Calculate total reward.
        if self.log_return:
            total_reward = math.log((total_portfolio + self.balance) / tmp_portfolio)
        else:
            total_reward = (total_portfolio + self.balance) - tmp_portfolio

        if done and self.use_final_reward:
            if self.log_return:
                total_reward = math.log(
                    (self.get_final_reward() + self.initial_balance) / self.initial_balance)
            else:
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

    def max_portfolio(self):
        r"""
        Calculate the max possible return value.
        The min value must appear at the day which before the max value.
        """
        best_min_value = None
        best_max_value = None
        best_return_value = 0
        for t in self.tickers:
            right = len(self.episode_data[t][self.target_feature]) - 1
            left = right - 1
            best_value = self.episode_data[t][self.target_feature][right] - self.episode_data[t][self.target_feature][left]
            # Start from the last day.
            while left >= 0:
                if self.episode_data[t][self.target_feature][left] > self.episode_data[t][self.target_feature][right]:
                    right = left
                    left = right - 1
                else:
                    left -= 1
                if left < 0:
                    break
                value = self.episode_data[t][self.target_feature][right] - self.episode_data[t][self.target_feature][left]
                if value > best_value:
                    best_value = value
                    if value > best_return_value:
                        best_return_value = value
                        best_min_value = self.episode_data[t][self.target_feature][left]
                        best_max_value = self.episode_data[t][self.target_feature][right]

        if best_return_value == 0:
            return self.initial_balance
        else:
            return self.initial_balance * (best_max_value / best_min_value)



