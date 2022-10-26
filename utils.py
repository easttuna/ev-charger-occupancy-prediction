import numpy as np
import pandas as pd


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	"""station should be in axis0 (index), timestamp should be axis1 (columns)

	Args:
		sequences (_type_): _description_
		n_steps_in (_type_): _description_
		n_steps_out (_type_): _description_

	Returns:
		_type_: _description_
	"""
	size = sequences.shape[1]
	xs = np.empty((0,n_steps_in))
	ys = np.empty((0,n_steps_out))

	for idx in range(1008-n_steps_in, size - (n_steps_in + n_steps_out)):
		x = sequences[:,idx:idx+n_steps_in]
		xs = np.vstack([xs, x])
		y = sequences[:, idx+n_steps_in:idx+n_steps_in+n_steps_out]
		ys = np.vstack([ys, y])
	return xs, ys


def station_features(station_array, station_df, n_windows):
	df = pd.DataFrame(data=station_array, columns=['station_name']).merge(station_df[['station_name', 'dcode']], how='left', on='station_name')
	name_encoder = {name:idx for idx, name in enumerate(df.station_name.unique())}
	dcode_encoder = {name:idx for idx, name in enumerate(df.dcode.unique())}

	df.station_name = df.station_name.map(name_encoder)
	df.dcode = df.dcode.map(dcode_encoder)

	return np.tile(df.values, (n_windows,1))


def time_features(time_idx, n_steps_in, n_steps_out, n_stations):
	df = pd.DataFrame(data=pd.to_datetime(time_idx), columns=['time'])
	df['t_index']  = df['time'].dt.hour.multiply(60).add(df['time'].dt.minute).floordiv(30)
	df['dow'] = df['time'].dt.dayofweek
	df['weekend'] = df.dow.isin([5,6]).astype(np.int64)
	del df['time']

	ts = np.empty((0,n_steps_out,3))
	for idx in range(1008-n_steps_in, len(time_idx) - (n_steps_in + n_steps_out)):
		t = df.values[np.newaxis, idx+n_steps_in:idx+n_steps_in+n_steps_out, :]
		ts = np.vstack([ts, t])

	return np.repeat(ts, n_stations, axis=0)


def history_sequences(sequences, n_steps_in, n_steps_out):
	size = sequences.shape[1]
	hs = np.empty((0,3))

	for idx in range(1008-n_steps_in, size - (n_steps_in + n_steps_out)):
		h = sequences[:, [idx+n_steps_in-1008, idx+n_steps_in-672, idx+n_steps_in-336]]
		hs = np.vstack([hs, h])
	return hs


if __name__ == '__main__':
    pass
