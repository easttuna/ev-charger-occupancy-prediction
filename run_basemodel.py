import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from utils import split_sequences, time_features, station_features, linear_split, stationwise_split, EvcFeatureGenerator
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
    np.random.seed(42)  # fix random seed

    sequences = pd.read_csv('./data/input_table/history_by_station_pub.csv', parse_dates=['time'])
    station_attributes = pd.read_csv('./data/input_table/pubstation_feature_scaled.csv')
    station_embeddings = pd.read_csv('./data/input_table/pubstation_umap-embedding.csv')

    feature_generator = EvcFeatureGenerator(sequences, station_attributes, station_embeddings)
    feature_generator.historic_seq_smoothing()  # smoothing 적용
    feature_generator.discretize_sequences()  # 이산화 (binary)
    feature_generator.slice_data(prob=0.9)  # mean availability 0.9 이하 선택

    PRED_STEP = 2
    R_seq, H_seq, T, S, Y = feature_generator.generate_features(n_in=12, n_out=6, n_hist=4, pred_step=PRED_STEP)
    print(feature_generator.n_stations)

    # 4) Train : Valid Split   
    # train_arrays, valid_arrays = linear_split(R_seq, H_seq, T, S, Y, train_frac=0.9)
    train_arrays, valid_arrays = stationwise_split(R_seq, H_seq, T, S, Y, n_station=feature_generator.n_stations, train_frac=0.9)
    trainset = EvcDataset(*train_arrays)
    validset = EvcDataset(*valid_arrays)
    print(f'Trainset Size: {len(trainset)}, Validset Size: {len(validset)}')

    # 5)  Data Loader
    # with negative over sampling
    OS_RATE = 5
    weights = np.where(trainset[:][-1].flatten() == 0., OS_RATE, 1)  # 5배
    num_samples = len(trainset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
    train_loader = DataLoader(trainset, batch_size=32, sampler=sampler)

    # without sampling
    # train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=1024, shuffle=False)

    START_POINT = 1004
    N_DAYS = 3
    sample_idx = [START_POINT + i*feature_generator.realtime_sequences.shape[0] for i in range(72*N_DAYS)]
    sample_data = validset[sample_idx]

    models = {'HistoricBase':HistoricBase, 'RealtimeBase':RealtimeBase, 'MultiSeqBase':MultiSeqBase, 'MultiSeqHybrid':MultiSeqHybrid, 
    'MultiSeqUmap':MultiSeqUmap, 'MultiSeqUmapEmbonly':MultiSeqUmapEmbonly}
    # models = {'MultiSeqBase':MultiSeqBase, 'MultiSeqHybrid':MultiSeqHybrid, 'MultiSeqUmapEmbonly':MultiSeqUmapEmbonly}

    for name, basemodel in models.items():
        print(f'-------{name}-------')
        if name in ['MultiSeqUmap', 'MultiSeqUmapEmbonly']:
            model = basemodel(hidden_size=16, embedding_dim=8, pretrained_embedding=feature_generator.umap_embedding_vectors)
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
        plt.savefig(f'./images/out-{PRED_STEP}_os-{OS_RATE}_model-{name}_.png')
