# Description: Configuration for each experiment.
from dataclasses import dataclass, asdict
import json
import pathlib
from envs import FinanceEnv
import copy
from typing import Union, Dict

@dataclass
class ModelConfig:
    r"""
    Configuration for model and optimizer.
    """
    state_dim: int = 8
    action_dim: int = 1
    hidden_dim: int = 256
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    update_iteration: int = 10
    clip: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    has_continuous_action_space: bool = True
    tanh_action: bool = True
    use_GAE: bool = True
    fix_var_param: Union[Dict, None] = None

    def __dict__(self):
        return asdict(self)


class Config:
    r"""
    Configuration for each experiment.
    """
    def __init__(
        self,
        exp_name: str,
        model_config: ModelConfig,
        seed: int = 42,
        training_episode_num: int = 100,
        update_timestep: int = 2048,
        env_name: str = 'FinanceEnv',
        env_kwargs: dict = {
            'start_date': '2010-01-01',
            'end_date': '2020-12-31',
            'ticker': '2330.TW',
            'state_feature_names': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
            'initial_balance': 10000,
            'initial_stock': 0,
            'episode_size': 30,
            'target_feature': 'Adj Close',
            'extra_feature_dict': None,
        },
        test_env_kwargs: dict = {
            'start_date': '2020-12-31',
            'end_date': None,
            'ticker': '2330.TW',
            'state_feature_names': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
            'initial_balance': 10000,
            'initial_stock': 0,
            'episode_size': 30,
            'target_feature': 'Adj Close',
            'extra_feature_dict': None,
        },
        log_timestep: int = 10,
        save_timestep: int = 100,
        action_std_decay_timestep: int = 1000000,
        min_action_std: float = 0.2,
    ):
        self.exp_name = exp_name
        self.model_config = model_config
        self.seed = seed
        self.training_episode_num = training_episode_num
        self.update_timestep = update_timestep
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.save_timestep = save_timestep
        self.log_timestep = log_timestep
        self.action_std_decay_timestep = action_std_decay_timestep
        self.min_action_std = min_action_std

    def __dict__(self):
        return {
            'exp_name': self.exp_name,
            'model_config': self.model_config.__dict__(),
            'seed': self.seed,
            'training_episode_num': self.training_episode_num,
            'update_timestep': self.update_timestep,
            'env_name': self.env_name,
            'env_kwargs': self.env_kwargs,
            'save_timestep': self.save_timestep,
            'log_timestep': self.log_timestep,
            'action_std_decay_timestep': self.action_std_decay_timestep,
            'min_action_std': self.min_action_std,
        }

    def save_config(self):
        if self.env_kwargs is None or self.env_kwargs == {}:
            return

        dir_name = pathlib.Path('checkpoint') / pathlib.Path(self.exp_name)
        if not dir_name.exists():
            dir_name.mkdir(parents=True, exist_ok=True)

        tmp = self.env_kwargs['extra_feature_dict']
        del self.env_kwargs['extra_feature_dict']
        with open(dir_name / 'config.json', 'w') as f:
            json.dump(self.__dict__(), f, indent=4)

        self.env_kwargs['extra_feature_dict'] = tmp

    @classmethod
    def load_config(cls, dir_name: str):
        path = pathlib.Path(dir_name) / 'config.json'
        with path.open('r') as f:
            config = json.load(f)
        config = cls(**config)
        config.model_config = ModelConfig(**config.model_config)
        return config

