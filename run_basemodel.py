import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from utils import split_sequences, station_features, time_features, history_sequences
from dataset import EvcDataset
from models import BaseHybrid, Fastfusion, DcodeHybrid, NaiveModel, MeanModel


def train(model, train_dataloader, epoch):
    model.train()
    criterion = nn.MSELoss()
    for b_i, (R, H, T, S, y) in enumerate(train_dataloader):


        pred = model(R, H, T, S)
        loss = criterion(pred, y)

        
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

    print('\nTest dataset:  Loss: {:.4f}, Accuracy: {:.4f}, R2: {:.4f}'.format(loss, accuracy, r2))


if __name__ == '__main__':
    history = pd.read_csv('./data/input_table/history_by_station.csv', parse_dates=['time'])
    station = pd.read_csv('./data/input_table/station_info.csv')
    data = history.set_index('time').T.reset_index().rename(columns={'index':'station_name'})
    data = data[data.station_name.isin(station.station_name)].set_index('station_name')
    data = data[data.mean(axis=1).le(0.9)]
    print('generating inputs...')
    N_STEPS_IN = 12
    N_STEPS_OUT = 6

    n_windows = data.shape[1] - (N_STEPS_IN + N_STEPS_OUT)
    n_stations = data.shape[0]

    S = station_features(station_array=data.index, station_df=station, n_windows=n_windows)
    T = time_features(data.columns, N_STEPS_IN, N_STEPS_OUT, n_stations)
    R, Y = split_sequences(data.values, N_STEPS_IN, N_STEPS_OUT)
    H = history_sequences(data.values, N_STEPS_IN,N_STEPS_OUT)
    print('done!')

    OUTPUT_IDX = 0
    T = T[:,OUTPUT_IDX,:]
    Y = Y[:,OUTPUT_IDX, np.newaxis]
    H = H[:, :, np.newaxis]
    R = R[:, :, np.newaxis]

    VALID_FRAC = 0.1
    
    num_valid = int(data.shape[0] * VALID_FRAC * (data.shape[1] - 1008))
    print(data.shape, num_valid)

    trainset = EvcDataset(R[:-num_valid,], H[:-num_valid], T[:-num_valid,], S[:-num_valid,], Y[:-num_valid,])
    validset = EvcDataset(R[-num_valid:,], H[-num_valid:,], T[-num_valid:,], S[-num_valid:,], Y[-num_valid:,])

    print(len(trainset), len(validset))

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=1024)


    # model = BaseHybrid(hidden_size=32, station_embedding_dim=32, embedding_dim=16)
    model = MeanModel()
    # optim = torch.optim.Adam(model.parameters())

    N_EPOCH = 1
    for epoch in range(1,N_EPOCH+1):
        # train(model, train_loader, optim, epoch)
        train(model, train_loader, epoch)
        test(model, valid_loader)
        print()
