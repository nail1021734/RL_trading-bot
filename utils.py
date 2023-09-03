from typing import List, Dict, Union

class ExtraFeatureProcessor:
    def __init__(self, extra_feature_names: Union[List[str], None]):
        if extra_feature_names is None:
            extra_feature_names = []
        self.extra_feature_names = extra_feature_names
        self.last_K_value = {}
        self.last_D_value = {}
        self.general_functions = {
            'date': self.date,
        }
        self.functions = {
            'average': self.average,
            'moving_average': self.moving_average,
            'K_value': self.K_value,
            'D_value': self.D_value,
            'J_value': self.J_value,
            'last_K_value': self.add_last_K_value,
            'last_D_value': self.add_last_D_value,
            'last_J_value': self.add_last_J_value,
        }

    def get_extra_feature_num(self, env: "envs.MultiFinanceEnv"):
        general_feature_num = 0
        feature_num = 0
        for func_name in self.extra_feature_names:
            if func_name in self.general_functions:
                general_feature_num += 1
            elif func_name in self.functions:
                feature_num += 1
            else:
                raise NotImplementedError

        return general_feature_num, feature_num

    def process(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        # Do general_functions first.
        for func_name in self.extra_feature_names:
            if func_name in self.general_functions:
                self.general_functions[func_name](env, state_dict)
        # Do functions.
        for func_name in self.extra_feature_names:
            if func_name in self.functions:
                self.functions[func_name](env, state_dict)

    def mean(self, x: List):
        return sum(x) / len(x)

    def average(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        for t in env.tickers:
            if env.day == 0:
                state_dict[t]['average'] = env.episode_data[t][env.target_feature][env.day]
            else:
                state_dict[t]['average'] = self.mean(env.episode_data[t][env.target_feature][:env.day])

    def moving_average(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        for t in env.tickers:
            if env.day == 0:
                state_dict[t]['moving_average'] = env.episode_data[t][env.target_feature][env.day]
            else:
                state_dict[t]['moving_average'] = self.mean(env.episode_data[t][env.target_feature][max(0, env.day-5): env.day])

    def date(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        # Convert date to 1~365.
        date_in_year = env.data_date[env.begin_date+env.day]
        date_in_year = int(date_in_year.month)*30 + int(date_in_year.day)
        state_dict['date'] = date_in_year


    def RSV_value(
        self,
        env: "envs.MultiFinanceEnv",
        day: int,
        day_window=9,
    ):
        # Record RSV value.
        RSV_table = {}

        # Calculate RSV value.
        for t in env.tickers:
            last_days = env.data[t][env.target_feature][max(0, env.begin_date+day-day_window): env.begin_date+day]
            if len(last_days) == 0:
                RSV_table[t] = 50
            else:
                highest = max(last_days)
                lowest = min(last_days)
                if highest == lowest:
                    RSV_table[t] = 50
                else:
                    RSV_table[t] = 100 * (env.episode_data[t][env.target_feature][day] - lowest) / (highest - lowest)

        return RSV_table

    def K_value(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        # Calculate K value.
        RSV_table = self.RSV_value(env=env, day=env.day)
        for t in env.tickers:
            if env.day == 0:
                state_dict[t]['K_value'] = 50
            else:
                state_dict[t]['K_value'] = 2/3 * self.last_K_value[t] + 1/3 * RSV_table[t]
            self.last_K_value[t] = state_dict[t]['K_value']

    def D_value(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        # Calculate D value.
        for t in env.tickers:
            if env.day == 0:
                state_dict[t]['D_value'] = 50
            else:
                state_dict[t]['D_value'] = 2/3 * self.last_D_value[t] + 1/3 * state_dict[t]['K_value']
            self.last_D_value[t] = state_dict[t]['D_value']

    def J_value(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        # Calculate J value.
        for t in env.tickers:
            state_dict[t]['J_value'] = 3 * state_dict[t]['K_value'] - 2 * state_dict[t]['D_value']

    def add_last_K_value(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        if env.day == 0:
            for t in env.tickers:
                state_dict[t]['last_K_value'] = 50
                self.last_K_value[t] = 50
        else:
            for t in env.tickers:
                state_dict[t]['last_K_value'] = self.last_K_value[t]

    def add_last_D_value(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        if env.day == 0:
            for t in env.tickers:
                state_dict[t]['last_D_value'] = 50
                self.last_D_value[t] = 50
        else:
            for t in env.tickers:
                state_dict[t]['last_D_value'] = self.last_D_value[t]

    def add_last_J_value(
        self,
        env: "envs.MultiFinanceEnv",
        state_dict: Dict,
    ):
        for t in env.tickers:
            state_dict[t]['last_J_value'] = 3 * self.last_K_value[t] - 2 * self.last_D_value[t]
