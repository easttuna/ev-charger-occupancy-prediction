import numpy as np
import pandas as pd
from tqdm import tqdm


def split_sequences(sequences, n_steps_in, n_steps_out, n_history):
    """transforms target sequence table to historic/real-time/target sequence features)

    Args:
        sequences (np.array): 2-dimensional target feature array (stations on axis0, timestep on axis1)
        n_steps_in (int): length of real-time sequence feature
        n_steps_out (int): length of target sequence
        n_history (int): length of historic sequence

    Returns:
        _type_: _description_
    """
    size = sequences.shape[1]
    rs = np.empty((0,n_steps_in))
    hs = np.empty((0, n_steps_out, n_history))
    ys = np.empty((0,n_steps_out))

    for idx in range(n_history * 336 - n_steps_in, size - (n_steps_in + n_steps_out)):
        r = sequences[:,idx:idx+n_steps_in]
        rs = np.vstack([rs, r])
        
        h = sequences[:, [idx + n_steps_in + out_step - 336*hist_step for out_step in range(n_steps_out) for hist_step in range(n_history, 0, -1)]]
        h = h.reshape(-1,n_steps_out, n_history)
        hs = np.vstack([hs, h])

        y = sequences[:, idx+n_steps_in:idx+n_steps_in+n_steps_out]
        ys = np.vstack([ys, y])

    return rs, hs, ys


def time_features(time_idx, n_steps_in, n_steps_out, n_history, n_stations):
    df = pd.DataFrame(data=pd.to_datetime(time_idx), columns=['time'])
    df['t_index']  = df['time'].dt.hour.multiply(60).add(df['time'].dt.minute).floordiv(30)
    df['dow'] = df['time'].dt.dayofweek
    df['weekend'] = df.dow.isin([5,6]).astype(np.int64)
    df = df[['t_index', 'dow', 'weekend']]

    ts = np.empty((0,n_steps_out,3))
    for idx in range(n_history * 336 - n_steps_in, len(time_idx) - (n_steps_in + n_steps_out)):
        t = df.values[np.newaxis, idx+n_steps_in:idx+n_steps_in+n_steps_out, :]
        ts = np.vstack([ts, t])

    return np.repeat(ts, n_stations, axis=0)


def station_features(station_array, station_df, n_windows):
    df = pd.DataFrame(data=station_array, columns=['station_name']).merge(station_df[['station_name', 'dcode']], how='left', on='station_name')
    name_encoder = {name:idx for idx, name in enumerate(df.station_name.unique())}
    dcode_encoder = {name:idx for idx, name in enumerate(df.dcode.unique())}

    df.station_name = df.station_name.map(name_encoder)
    df.dcode = df.dcode.map(dcode_encoder)

    return np.tile(df.values, (n_windows,1))


if __name__ == '__main__':
    # check split sequence function
    charge = pd.read_csv('./data/input_table/history_by_station.csv', parse_dates=['time'])
    station = pd.read_csv('./data/input_table/station_info.csv')
    data = charge.set_index('time').T.reset_index().rename(columns={'index':'station_name'})
    data = data[data.station_name.isin(station.station_name)].set_index('station_name')
    data = data[:50]

    N_IN = 12
    N_OUT = 6
    N_HIST = 3

    print('Split Sequences..')
    R, H, Y = split_sequences(sequences=data.values, n_steps_in=N_IN, n_steps_out=N_OUT, n_history=N_HIST)
    print('Done!')
    print(R.shape, H.shape, Y.shape)

    print('Generate time features...')
    T = time_features(time_idx=data.columns, n_steps_in=N_IN, n_steps_out=N_OUT, n_history=N_HIST, n_stations=data.shape[0])
    print('Done!')
    print(T.shape)

    print('Generate station features...')
    S = station_features(station_array=data.index, station_df=station, n_windows=data.shape[1] - (N_OUT+336*N_HIST))
    print(S.shape)
