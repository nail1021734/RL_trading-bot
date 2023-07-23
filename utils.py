from typing import List, Dict

def mean(x: List):
    return sum(x) / len(x)
def average(
    tickers: List[str],
    episode_data: Dict,
    day: int,
    state_dict: Dict[str, float],
    state: List[float],
    target_feature: str,
):
    for ticker in tickers:
        if day == 0:
            state.append(episode_data[ticker][target_feature][day])
            state_dict['avg_'+ticker] = episode_data[ticker][target_feature][day]
        else:
            state.append(mean(episode_data[ticker][target_feature][:day]))
            state_dict['avg_'+ticker] = mean(episode_data[ticker][target_feature][:day])

def mov_average(
    tickers: List[str],
    episode_data: Dict,
    day: int,
    state_dict: Dict[str, float],
    state: List[float],
    target_feature: str,
):
    for ticker in tickers:
        if day == 0:
            state.append(episode_data[ticker][target_feature][day])
            state_dict['avg_'+ticker] = episode_data[ticker][target_feature][day]
        else:
            state.append(mean(episode_data[ticker][target_feature][max(0, day-5):day]))
            state_dict['avg_'+ticker] = mean(episode_data[ticker][target_feature][max(0, day-5):day])
