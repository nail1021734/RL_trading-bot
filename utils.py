from typing import List, Dict, Union

class ExtraFeatureProcessor:
    def __init__(self, extra_feature_names: Union[List[str], None]):
        if extra_feature_names is None:
            extra_feature_names = []
        self.extra_feature_names = extra_feature_names
        self.general_functions = {
            'date': self.date,
        }
        self.functions = {
            'average': self.average,
            'moving_average': self.moving_average,
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

