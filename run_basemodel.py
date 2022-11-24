import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from utils import split_sequences, station_features, time_features, EvcFeatureGenerator
from dataset import EvcDataset
from basemodels import HistoricBase, RealtimeBase, MultiSeqBase, MultiSeqHybrid, MultiSeqUmap, MultiSeqUmapEmbonly


def train(model, train_dataloader, optim, epoch, verbose=0):
    model.train()
    criterion = nn.BCELoss()
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
    y_total = y_total.int().numpy()
    pred_total = pred_total.numpy()
    pred_label = np.where(pred_total > 0.5, 1, 0)


    recall = recall_score(y_total, pred_label)
    precision = precision_score(y_total, pred_label)
    f1 = f1_score(y_total, pred_label)
    accuracy = accuracy_score(y_total, pred_label)
    bal_accuracy = balanced_accuracy_score(y_total, pred_label)
    auc = roc_auc_score(y_total, pred_total)

    # print('Test dataset:  Loss: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}, Balanced-Accuracy: {:.4f}, AUC: {:.4f}' \
    # .format(loss, recall, precision, f1, accuracy, bal_accuracy, auc))
    print('Test dataset:  Loss: {:.4f}, Accuracy: {:.4f}, Balanced-Accuracy: {:.4f}, AUC: {:.4f}' \
    .format(loss, accuracy, bal_accuracy, auc))


if __name__ == '__main__':
    # 1) Load Data
    station_sequence = pd.read_csv('./data/input_table/history_by_station_pub.csv', parse_dates=['time'])
    station_attributes = pd.read_csv('./data/input_table/pubstation_feature_scaled.csv')
    station_embeddings = pd.read_csv('./data/input_table/pubstation_umap-embedding.csv')

    sid_encoder = {name:idx for idx, name in enumerate(station_embeddings.sid)}
    station_embeddings.sid = station_embeddings.sid.map(sid_encoder)
    station_attributes.sid = station_attributes.sid.map(sid_encoder)

    data = station_sequence.set_index('time').T.reset_index().rename(columns={'index':'sid'})
    data.sid = data.sid.map(sid_encoder)
    data = data[data.sid.isin(station_attributes.sid)].set_index('sid')  # station feature가 있는 데이터로 한정

    # smoothing
    historic_data = data.T
    historic_data.index = pd.to_datetime(historic_data.index)
    historic_data = historic_data.resample(rule='1h').mean().T

    # transforms targer var. to binary indicator (1:high availabiltity, 0: low availability)
    data = data.mask(lambda x: x < 0.5,  1).mask(lambda x: x != 1, 0)
    historic_data = historic_data.mask(lambda x: x < 0.5,  1).mask(lambda x: x != 1, 0)

    select_idx = data.mean(axis=1).le(0.9)
    data = data[select_idx]  # False 라벨이 10% 이상 존재하는 데이터 사용
    historic_data = historic_data[select_idx]
    print(data.shape)

    umap_embedding = torch.tensor(station_embeddings.drop(columns=['sid']).values).float()

    # 2) Feature Generation
    print('generating inputs...')
    N_IN = 12
    N_OUT = 6
    N_HIST = 4

    n_stations = data.shape[0]
    n_windows = data.shape[1] - (N_OUT + 504*N_HIST)
    R_seq, H_seq, Y_seq = split_sequences(sequences=data.values, 
                                          n_steps_in=N_IN, n_steps_out=N_OUT, n_history=N_HIST, 
                                          historic_sequences=np.repeat(historic_data.values, 3, 1))
    T = time_features(time_idx=data.columns, n_steps_in=N_IN, n_steps_out=N_OUT, n_history=N_HIST, n_stations=n_stations)
    S = station_features(station_array=data.index, station_df=station_attributes, n_windows=n_windows) 
    print('done!')

    # 3) Set Dimension
    R_seq = R_seq[:, :, np.newaxis]
    OUTPUT_IDX = 1
    H_seq = H_seq[:, OUTPUT_IDX, :, np.newaxis]
    T = T[:,OUTPUT_IDX,:]
    Y = Y_seq[:,OUTPUT_IDX, np.newaxis]

    # 4) Train : Valid Split
    TRAIN_FRAC = 0.9
    n_train = int(data.shape[0] * TRAIN_FRAC * n_windows)
    trainset = EvcDataset(R_seq[:n_train,], H_seq[:n_train], T[:n_train,], S[:n_train,], Y[:n_train,])
    validset = EvcDataset(R_seq[n_train:,], H_seq[n_train:,], T[n_train:,], S[n_train:,], Y[n_train:,])
    print(f'Trainset Size: {len(trainset)}, Validset Size: {len(validset)}')

    # 5)  Data Loader
    # with negative over sampling
    OS_RATE = 5
    weights = np.where(trainset[:][-1].flatten() == 0., OS_RATE, 1)  # 9배
    num_samples = len(trainset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
    train_loader = DataLoader(trainset, batch_size=32, sampler=sampler)

    # without sampling
    # train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=1024, shuffle=False)

    START_POINT = 1004
    N_DAYS = 3
    sample_idx = [START_POINT + i*n_stations for i in range(72*N_DAYS)]
    sample_data = validset[sample_idx]

    models = {'HistoricBase':HistoricBase, 'RealtimeBase':RealtimeBase, 'MultiSeqBase':MultiSeqBase, 'MultiSeqHybrid':MultiSeqHybrid, 
    'MultiSeqUmap':MultiSeqUmap, 'MultiSeqUmapEmbonly':MultiSeqUmapEmbonly}
    # models = {'MultiSeqBase':MultiSeqBase, 'MultiSeqHybrid':MultiSeqHybrid, 'MultiSeqUmapEmbonly':MultiSeqUmapEmbonly}

    for name, basemodel in models.items():
        print(f'-------{name}-------')
        if name in ['MultiSeqUmap', 'MultiSeqUmapEmbonly']:
            model = basemodel(hidden_size=16, embedding_dim=8, pretrained_embedding=umap_embedding)
        else:
            model = basemodel(hidden_size=16, embedding_dim=8)
        optim = torch.optim.Adam(model.parameters())

        N_EPOCH = 10
        for epoch in range(1,N_EPOCH+1):
            print(f'<<Epoch {epoch}>>', end='\t')
            train(model, train_loader, optim, epoch, verbose=0)
            test(model, valid_loader)

        y_true = sample_data[4].flatten().detach().numpy()
        y_pred = model(*sample_data[:4]).flatten().detach().numpy()
        y_pred = np.where(y_pred > 0.5, 1, 0)

        fig, ax = plt.subplots(figsize=(40,8))
        ax.plot(y_true, color='g')
        ax.plot(y_pred, color='r')
        plt.savefig(f'./images/out-{OUTPUT_IDX}_os-{OS_RATE}_{name}_result_smooth.png')
