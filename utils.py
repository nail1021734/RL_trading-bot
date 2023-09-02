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
    begin_date: int,
    data_date: List[str],
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
    begin_date: str,
    data_date: List[str],
):
    for ticker in tickers:
        if day == 0:
            state.append(episode_data[ticker][target_feature][day])
            state_dict['avg_'+ticker] = episode_data[ticker][target_feature][day]
        else:
            state.append(mean(episode_data[ticker][target_feature][max(0, day-5):day]))
            state_dict['avg_'+ticker] = mean(episode_data[ticker][target_feature][max(0, day-5):day])


def date(
    tickers: List[str],
    episode_data: Dict,
    day: int,
    state_dict: Dict[str, float],
    state: List[float],
    target_feature: str,
    begin_date: str,
    data_date: List[str],
):
    # Convert date to 1~365.
    date_in_year = data_date[begin_date+day]
    date_in_year = int(date_in_year.month)*30 + int(date_in_year.day)
    state.append(date_in_year)
    state_dict['date'] = date_in_year

