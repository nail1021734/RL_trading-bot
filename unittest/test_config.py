import os
import pytest
from config import Config, ModelConfig
import pathlib
from envs import FinanceEnv


def test_ModelConfig():
    mConfig = ModelConfig(
        algorithm='PPO',
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        hidden_dim=256,
        gamma=0.99,
        noise_var=0.1,
        states_dim=8,
        actions_dim=1,
        clip=0.2,
        time_step_per_batch=2048,
        max_time_step_per_episode=1000,
        n_updates_per_iteration=10,
        vf_coef=0.5,
    )

    assert mConfig.algorithm == 'PPO'
    assert mConfig.actor_learning_rate == 1e-4
    assert mConfig.critic_learning_rate == 1e-3
    assert mConfig.hidden_dim == 256
    assert mConfig.gamma == 0.99
    assert mConfig.noise_var == 0.1
    assert mConfig.states_dim == 8
    assert mConfig.actions_dim == 1
    assert mConfig.clip == 0.2
    assert mConfig.time_step_per_batch == 2048
    assert mConfig.max_time_step_per_episode == 1000
    assert mConfig.n_updates_per_iteration == 10
    assert mConfig.vf_coef == 0.5

    assert mConfig.__dict__() == {
        'algorithm': 'PPO',
        'actor_learning_rate': 1e-4,
        'critic_learning_rate': 1e-3,
        'hidden_dim': 256,
        'gamma': 0.99,
        'noise_var': 0.1,
        'states_dim': 8,
        'actions_dim': 1,
        'clip': 0.2,
        'time_step_per_batch': 2048,
        'max_time_step_per_episode': 1000,
        'n_updates_per_iteration': 10,
        'vf_coef': 0.5,
    }



def test_Config():
    mConfig = ModelConfig(
        algorithm='PPO',
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        hidden_dim=256,
        gamma=0.99,
        noise_var=0.1,
        states_dim=8,
        actions_dim=1,
        clip=0.2,
        time_step_per_batch=2048,
        max_time_step_per_episode=1000,
        n_updates_per_iteration=10,
        vf_coef=0.5,
    )

    config = Config(
        exp_name='test',
        model_config=mConfig,
        seed=42,
        batch_size=128,
        num_episode=100,
        env_name='FinanceEnv',
        env_kwargs={
            'start_date': '2010-01-01',
            'end_date': '2020-12-31',
            'ticker': 'AAPL',
            'feature_list': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'window_size': 10,
            'initial_balance': 1000000,
        },
        save_step=100,
    )

    assert config.exp_name == 'test'
    assert config.seed == 42
    assert config.batch_size == 128
    assert config.num_episode == 100
    assert config.env_name == 'FinanceEnv'
    assert config.env_kwargs == {
        'start_date': '2010-01-01',
        'end_date': '2020-12-31',
        'ticker': 'AAPL',
        'feature_list': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'window_size': 10,
        'initial_balance': 1000000,
    }
    assert config.save_step == 100
    assert config.model_config == mConfig

    assert config.__dict__() == {
        'exp_name': 'test',
        'seed': 42,
        'batch_size': 128,
        'num_episode': 100,
        'env_name': 'FinanceEnv',
        'env_kwargs': {
            'start_date': '2010-01-01',
            'end_date': '2020-12-31',
            'ticker': 'AAPL',
            'feature_list': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'window_size': 10,
            'initial_balance': 1000000,
        },
        'save_step': 100,
        'model_config': {
            'algorithm': 'PPO',
            'actor_learning_rate': 1e-4,
            'critic_learning_rate': 1e-3,
            'hidden_dim': 256,
            'gamma': 0.99,
            'noise_var': 0.1,
            'states_dim': 8,
            'actions_dim': 1,
            'clip': 0.2,
            'time_step_per_batch': 2048,
            'max_time_step_per_episode': 1000,
            'n_updates_per_iteration': 10,
            'vf_coef': 0.5,
        }
    }


def test_Config_save_and_load_config():
    mConfig = ModelConfig(
        algorithm='PPO',
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        hidden_dim=256,
        gamma=0.99,
        noise_var=0.1,
        states_dim=8,
        actions_dim=1,
        clip=0.2,
        time_step_per_batch=2048,
        max_time_step_per_episode=1000,
        n_updates_per_iteration=10,
    )

    config = Config(
        exp_name='test',
        model_config=mConfig,
        seed=42,
        batch_size=128,
        num_episode=100,
        env_name='FinanceEnv',
        env_kwargs={
            'start_date': '2010-01-01',
            'end_date': '2020-12-31',
            'ticker': 'AAPL',
            'feature_list': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'window_size': 10,
            'initial_balance': 1000000,
        },
        save_step=100,
    )

    # Testing save_config.
    config.save_config()
    assert (pathlib.Path(config.exp_name)/'config.json').exists()

    # Testing load_config.
    config = Config.load_config(config.exp_name)
    assert config.exp_name == 'test'
    assert config.seed == 42
    assert config.batch_size == 128
    assert config.num_episode == 100
    assert config.env_name == 'FinanceEnv'
    assert config.env_kwargs == {
        'start_date': '2010-01-01',
        'end_date': '2020-12-31',
        'ticker': 'AAPL',
        'feature_list': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'window_size': 10,
        'initial_balance': 1000000,
    }
    assert config.save_step == 100
    assert config.model_config == mConfig

    (pathlib.Path(config.exp_name)/'config.json').unlink()
    os.rmdir(config.exp_name)

