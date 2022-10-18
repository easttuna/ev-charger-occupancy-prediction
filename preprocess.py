import numpy as np
import pandas as pd


def extract_occupancy_cls(cid_df, start_date, end_date, interval_min):
    """transform charge log table to occupancy-by-window table for each charging station id

    Args:
        cid_df (pd.DataFrame): charging station log table of one charging stationID
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        interval_min (int): window size (minute)

    Returns:
        pd.Series: Returns occupancy at each point in time index
    """
    n_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    time_idx = pd.date_range(start=start_date,
                             freq=f'{interval_min}min',
                             periods=n_days*24*60//interval_min)

    result = np.full(time_idx.shape[0], False)
    for start_time, finish_time in zip(cid_df.start_time, cid_df.finish_time):
        occupied = (time_idx >= start_time) & (time_idx <= finish_time)
        result = result | occupied
    return pd.Series(data=result, index=time_idx, dtype='Int32')


def extract_occupancy_reg(cid_df, start_date, end_date, interval_min):
    """transform charge log table to occupancy-by-window table for each charging station id

    Args:
        cid_df (pd.DataFrame): charging station log table of one charging stationID
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        interval_min (int): window size (minute)

    Returns:
        pd.Series: Returns the occupancy rate for each period
    """
    n_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    time_idx = pd.date_range(start=start_date, freq='1min', periods=n_days*24*60)
    
    start = pd.Series(index=cid_df.start_time, data=1, name='start').resample('1min').sum()
    finish = pd.Series(index=cid_df.finish_time, data=1, name='finish').resample('1min').sum()
    
    occupancy = pd.Series(index=time_idx, name='occupied', data=0).to_frame() \
        .join(start, how='left').join(finish, how='left')
    occupancy.start = occupancy.start.replace({1.:2.}).fillna(0)
    occupancy.finish = occupancy.finish.fillna(0)
    occupancy.occupied = occupancy.start.add(occupancy.finish).replace({0.:np.nan}).ffill().sub(1).fillna(0).astype('Int32')

    return occupancy.resample(f'{interval_min}min').occupied.sum()


if __name__ == '__main__':
    pass
