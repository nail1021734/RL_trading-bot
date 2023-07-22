from typing import List, Dict
from pandas import DataFrame

def average(
    tickers: List[str],
    episode_data: DataFrame,
    day: int,
    state_dict: Dict[str, float],
    state: List[float],
    target_feature: str,
):
    for ticker in tickers:
        if day == 0:
            state.append(episode_data.iloc[day][target_feature][ticker])
            state_dict['avg_'+ticker] = episode_data.iloc[day][target_feature][ticker]
        else:
            state.append(episode_data.iloc[:day][target_feature][ticker].mean())
            state_dict['avg_'+ticker] = episode_data.iloc[:day][target_feature][ticker].mean()
