import numpy as np
import pandas as pd


def log_to_occupancy(cid_df, start_date, end_date, interval_min):
    """transform charge log table to occupancy-by-window table for each charging station id

    Args:
        cid_df (pd.DataFrame): charging station log table of one charging stationID
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        interval_min (int): window size (minute)
    """
    n_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    time_idx = pd.date_range(start=start_date,
                             freq=f'{interval_min}min',
                             periods=n_days*24*60//120)

    occupancy = np.full(time_idx.shape[0], False)
    for start_time, finish_time in zip(cid_df.start_time, cid_df.finish_time):
        occupied = (time_idx >= start_time) & (time_idx <= finish_time)
        occupancy = occupancy | occupied
    return pd.Series(data=occupancy, index=time_idx, dtype='Int32')
