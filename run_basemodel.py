import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from utils import split_sequences, station_features, time_features
from dataset import EvcDataset
from models import BaseHybrid, GatingHybrid
from basemodels import HistoricBase, RealtimeBase, MultiSeqBase, GatingSeqBase, GatingSeqEmbedding


def train(model, train_dataloader, optim, epoch, verbose=0):
    model.train()
    criterion = nn.MSELoss()
    for b_i, (R, H, T, S, y) in enumerate(train_dataloader):
        optim.zero_grad()
        pred = model(R, H, T, S)
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        
        if verbose:
            if b_i % 3000 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i * len(R), len(train_dataloader.dataset),
                    100 * b_i / len(train_dataloader), loss.item()
                ))

def test(model, test_dataloader):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    loss = 0

    with torch.no_grad():
        pred_total = torch.Tensor()
        y_total = torch.Tensor()

        for R, H, T, S, y in test_dataloader:
            pred = model(R, H, T, S)
            loss += criterion(pred, y).item()
            pred_total = torch.cat((pred_total, pred.flatten()), dim=0)
            y_total = torch.cat((y_total, y.flatten()), dim=0)

    loss /= len(test_dataloader.dataset)
    error = y_total - pred_total
    accuracy = 1- (torch.norm(error) / torch.norm(y_total))
    r2 = r2_score(y_total, pred_total)

    print('Test dataset:  Loss: {:.4f}, Accuracy: {:.4f}, R2: {:.4f}'.format(loss, accuracy, r2))


if __name__ == '__main__':
    history = pd.read_csv('./data/input_table/history_by_station.csv', parse_dates=['time'])
    station = pd.read_csv('./data/input_table/station_info.csv')
    data = history.set_index('time').T.reset_index().rename(columns={'index':'station_name'})
    data = data[data.station_name.isin(station.station_name)].set_index('station_name')
    data = data[data.mean(axis=1).le(0.9)]

    print('generating inputs...')
    N_STEPS_IN = 12
    N_STEPS_OUT = 6
    N_HISTORY = 4

    n_stations = data.shape[0]
    n_windows = data.shape[1] - (N_STEPS_OUT + 336*N_HISTORY)
    R, H, Y = split_sequences(data.values, N_STEPS_IN, N_STEPS_OUT, N_HISTORY)
    T = time_features(data.columns, N_STEPS_IN, N_STEPS_OUT, N_HISTORY, n_stations)
    S = station_features(station_array=data.index, station_df=station, n_windows=n_windows) 
    print('done!')

    R = R[:, :, np.newaxis]
   
    OUTPUT_IDX = 1
    H = H[:, OUTPUT_IDX, :, np.newaxis]
    T = T[:,OUTPUT_IDX,:]
    Y = Y[:,OUTPUT_IDX, np.newaxis]

    VALID_FRAC = 0.1
    num_valid = int(data.shape[0] * VALID_FRAC * n_windows)

    trainset = EvcDataset(R[:-num_valid,], H[:-num_valid], T[:-num_valid,], S[:-num_valid,], Y[:-num_valid,])
    validset = EvcDataset(R[-num_valid:,], H[-num_valid:,], T[-num_valid:,], S[-num_valid:,], Y[-num_valid:,])
    print(f'Trainset Size: {len(trainset)}, Validset Size: {len(validset)}')
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=1024)

    START_POINT = 42
    N_DAYS = 3
    sample_idx = [START_POINT + i*n_stations for i in range(48*N_DAYS)]
    sample_data = validset[sample_idx]

    models = {'HistoricBase':HistoricBase, 'RealtimeBase':RealtimeBase, 'MultiSeqBase':MultiSeqBase, 'GatingSeqBase':GatingSeqBase, 'GatingSeqEmbedding':GatingSeqEmbedding}
    for name, basemodel in models.items():
        print(f'-------{name}-------')
        model = basemodel(hidden_size=16, embedding_dim=8)
        optim = torch.optim.Adam(model.parameters())

        N_EPOCH = 5
        for epoch in range(1,N_EPOCH+1):
            print(f'<<Epoch {epoch}>>', end='\t')
            train(model, train_loader, optim, epoch, 0)
            test(model, valid_loader)

        y_true = sample_data[4].flatten()
        y_pred = model(*sample_data[:4]).flatten()

        fig, ax = plt.subplots(figsize=(40,8))
        ax.plot(y_true.detach().numpy(), color='g')
        ax.plot(y_pred.detach().numpy(), color='r')
        plt.savefig(f'./images/{name}_out-{OUTPUT_IDX}_result.png')
