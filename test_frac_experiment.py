import argparse
import time
import os.path as osp
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from utils import linear_split, stationwise_split, EvcFeatureGenerator
from dataset import EvcDataset
from basemodels import HistoricBase, RealtimeBase, MultiSeqBase, MultiSeqHybrid, MultiSeqUmap, MultiSeqHybridUmap, MultiSeqUmapGating, MultiSeqUmapEarlyGating


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
            if b_i % 5_000 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i * len(R), len(train_dataloader.dataset),
                    100 * b_i / len(train_dataloader), loss.item()
                ))

def test(model, test_dataloader):
    model.eval()
    criterion = nn.BCELoss(reduction='sum')
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
    specificity = recall_score(y_total, pred_label, pos_label=0)
    precision = precision_score(y_total, pred_label)
    f1 = f1_score(y_total, pred_label)
    accuracy = accuracy_score(y_total, pred_label)
    bal_accuracy = balanced_accuracy_score(y_total, pred_label)
    auc = roc_auc_score(y_total, pred_total)

    print('Test dataset:  Loss: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Precision: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}, Balanced-Accuracy: {:.4f}, AUC: {:.4f}' \
    .format(loss, recall, specificity, precision, f1, accuracy, bal_accuracy, auc))


def run(args):
    np.random.seed(42)  # fix random seed
    data_dir = osp.join('./data/input_table/by_test_frac/', args.test_frac)

    train_seq = pd.read_csv(osp.join(data_dir, 'train_sequences_cls.csv'), parse_dates=['time'])
    test_seq = pd.read_csv(osp.join(data_dir, 'test_sequences_cls.csv'), parse_dates=['time'])

    station_attributes = pd.read_csv(osp.join(data_dir, 'station_attributes.csv'))
    station_embeddings = pd.read_csv(osp.join(data_dir, 'station_embedding.csv'))
    train_generator = EvcFeatureGenerator(train_seq, station_attributes.copy(), station_embeddings.copy())
    test_generator = EvcFeatureGenerator(test_seq, station_attributes.copy(), station_embeddings.copy())

    if args.history_smoothing:
        train_generator.historic_seq_smoothing()  # smoothing 적용
        test_generator.historic_seq_smoothing()
    # train_generator.discretize_sequences()  # 이산화 (binary)
    # test_generator.discretize_sequences()
    # train_generator.slice_data(prob=0.9)  # mean availability 0.9 이하 선택

    start = time.time()
    trainset = EvcDataset(
        *train_generator.generate_features(n_in=args.n_in, n_out=args.n_out, n_hist=args.n_hist, pred_step=args.pred_step)
    )
    testset = EvcDataset(
        *test_generator.generate_features(n_in=args.n_in, n_out=args.n_out, n_hist=args.n_hist, pred_step=args.pred_step)
    )
    end = time.time()
    print(f'took {(end-start)/60:.0f}mins generating features')
    print('number of train stations: ', train_generator.n_stations)
    print('number of test stations: ', test_generator.n_stations)

    # 5)  Data Loader
    # with negative over sampling
    
    pos_ratio = train_generator.realtime_sequences.mean(axis=1).mean()
    os_rate = pos_ratio / (1-pos_ratio)
    weights = np.where(trainset[:][-1].flatten() == 0., os_rate, 1)  # trainset 배합에 따라 결정
    num_samples = len(trainset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
    
    train_loader = DataLoader(trainset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(testset, batch_size=1024, shuffle=False)

    START_POINT = 10
    N_DAYS = 3
    sample_idx = [START_POINT + i*test_generator.n_stations for i in range(72*N_DAYS)]
    sample_data = testset[sample_idx]

    models = {'MultiSeqBase':MultiSeqBase, 'MultiSeqUmap':MultiSeqUmap, 'MultiSeqHybrid':MultiSeqHybrid, 'MultiSeqUmapEarlyGating':MultiSeqUmapEarlyGating}


    for name, basemodel in models.items():
        print(f'-------{name}-------')
        if name in ['MultiSeqUmap', 'MultiSeqHybridUmap', 'MultiSeqUmapGating', 'MultiSeqUmapEarlyGating']:
            model = basemodel(hidden_size=32, embedding_dim=4, pretrained_embedding=train_generator.umap_embedding_vectors, dropout_p=args.dropout_p, freeze_embedding=True)
        else:
            model = basemodel(hidden_size=32, embedding_dim=4, dropout_p=args.dropout_p)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, args.n_epoch + 1):
            print(f'<<Epoch {epoch}>>', end='\n')
            train(model, train_loader, optim, epoch, verbose=1)
            test(model, test_loader)

        y_true = sample_data[4].flatten().detach().numpy()
        y_pred = model(*sample_data[:4]).flatten().detach().numpy()
        y_pred = np.where(y_pred > 0.5, 1, 0)

        fig, ax = plt.subplots(figsize=(40,8))
        ax.plot(y_true, color='g')
        ax.plot(y_pred, color='r')
        plt.savefig(f'./images/testfrac-{args.test_frac}_predstep-{args.pred_step}_model-{name}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for running model comparison')
    
    parser.add_argument('--test_frac', default='0.5', type=str)
    parser.add_argument('--n_in', default=12, type=int)
    parser.add_argument('--n_out', default=6, type=int)
    parser.add_argument('--n_hist', default=4, type=int)
    parser.add_argument('--history_smoothing', action='store_true')
    parser.add_argument('--pred_step', default=6, type=int)
    parser.add_argument('--dropout_p', default=0.2, type=float)
    parser.add_argument('--n_epoch', default=5, type=int)
    args = parser.parse_args()

    run(args)