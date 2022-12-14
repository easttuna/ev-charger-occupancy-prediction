import argparse
import time
import os.path as osp
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score

from utils import linear_split, stationwise_split, EvcFeatureGenerator
from dataset import EvcDataset
from basemodels_multiclass import MultiSeqBase, MultiSeqNormal, MultiSeqUmap, MultiSeqUmapEmb, MultiSeqUmapEmbGating


def train(model, train_dataloader, optim, epoch, verbose=0):
    model.train()
    for b_i, (R, H, T, S, y) in enumerate(train_dataloader):
        optim.zero_grad()
        pred = model(R, H, T, S)
        loss = F.nll_loss(pred, y.flatten())
        loss.backward()
        optim.step()
        
        if verbose:
            if b_i % 5000 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i * len(R), len(train_dataloader.dataset),
                    100 * b_i / len(train_dataloader), loss.item()
                ))

def test(model, test_dataloader):
    model.eval()

    loss = 0
    with torch.no_grad():
        pred_total = torch.Tensor()
        y_total = torch.Tensor()

        for R, H, T, S, y in test_dataloader:
            pred = model(R, H, T, S)
            loss += F.nll_loss(pred, y.flatten(), reduction='sum').item()
            pred_total = torch.cat((pred_total, pred), dim=0)
            y_total = torch.cat((y_total, y), dim=0)

    # metrics
    loss /= len(test_dataloader.dataset)
    pred_labels = pred_total.argmax(dim=1, keepdim=True)
    pred_exp = torch.exp(pred_total).detach().numpy()

    f1_macro = f1_score(y_total, pred_labels, average='macro')
    f1_micro = f1_score(y_total, pred_labels, average='micro')
    f1_weighted = f1_score(y_total, pred_labels, average='weighted')

    auc_macro = roc_auc_score(y_total.flatten(), pred_exp, average='macro', multi_class='ovr')
    auc_weighted = roc_auc_score(y_total.flatten(), pred_exp, average='weighted', multi_class='ovr')

    recall = recall_score(y_total, pred_labels, labels=[0,1,2], average=None)
    accuracy = accuracy_score(y_total, pred_labels)
    bal_accuracy = balanced_accuracy_score(y_total, pred_labels)

    print('Test dataset:  Loss: {:.4f}, F1: [macro: {:.4f}, micro: {:.4f}, weighted: {:.4f}], Accuracy: {:.4f}, Recalls: [0: {:.2f}, 1: {:.2f}, 2: {:.2f}], Balanced-Accuracy: {:.4f}, AUC: [macro: {:.4f}, weighted: {:.4f}]' \
    .format(loss, f1_macro, f1_micro, f1_weighted, accuracy, recall[0], recall[1], recall[2], bal_accuracy, auc_macro, auc_weighted))

    metrics = {'loss':loss, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'f1_weighted':f1_weighted, 
               'auc_macro':auc_macro, 'auc_weighted':auc_weighted, 'bal_accuracy':bal_accuracy, 'accuracy':accuracy,
               'recall_0':recall[0], 'recall_1':recall[1], 'recall_2':recall[2]}

    return metrics


def run(args):
    np.random.seed(42)  # fix random seed
    data_dir = osp.join('./data/input_table/final_data')

    train_seq = pd.read_csv(osp.join(data_dir, 'total_sequences_multicls_size20_bin3_dim8.csv'), parse_dates=['time'])
    station_attributes = pd.read_csv(osp.join(data_dir, 'station_attributes_v2.csv'))
    station_embeddings = pd.read_csv(osp.join(data_dir, 'station_embedding_v2.csv'))
    train_generator = EvcFeatureGenerator(train_seq, station_attributes.copy(), station_embeddings.copy())

    print(train_seq.shape, station_attributes.shape, station_embeddings.shape)

    if args.history_smoothing:
        # train_generator.historic_seq_smoothing()  # smoothing 적용
        # test_generator.historic_seq_smoothing()
        pass

    start = time.time()
    R_seq, H_seq, T, S, Y = train_generator.generate_features(n_in=args.n_in, n_out=args.n_out, n_hist=args.n_hist, pred_step=args.pred_step)
    end = time.time()
    print(f'took {(end-start)/60:.0f}mins generating features')
    print('number of train stations: ', train_generator.n_stations)

    # 4) Train : Valid Split
    train_arrays, valid_arrays = linear_split(R_seq, H_seq, T, S, Y, n_station=train_generator.n_stations, train_frac=(1-float(args.test_frac)))
    trainset = EvcDataset(*train_arrays)
    testset = EvcDataset(*valid_arrays)
    print(f'Trainset Size: {len(trainset)}, Testset Size: {len(testset)}')


    # 5)  Data Loader
    # with negative over sampling
    labels = trainset[:][-1].flatten().numpy()
    label_counter = Counter(labels)
    weight_encoder = {l:(1_000/cnt) for l,cnt in label_counter.items()}
    weights = list(map(weight_encoder.get, labels))
    num_samples = len(trainset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
    train_loader = DataLoader(trainset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(testset, batch_size=1024, shuffle=False)

    models = {'MultiSeqBase':MultiSeqBase, 'MultiSeqNormal':MultiSeqNormal, 'MultiSeqUmap':MultiSeqUmap, 
              'MultiSeqUmapEmb':MultiSeqUmapEmb, 'MultiSeqUmapEmbGating':MultiSeqUmapEmbGating}

    EMB_MODELS = ['MultiSeqUmap', 'MultiSeqUmapEmb', 'MultiSeqUmapEmbGating']

    result_metrics = []
    for name, basemodel in models.items():
        print(f'-------{name}-------')
        if name in EMB_MODELS:
            model = basemodel(n_labels=3, hidden_size=32, embedding_dim=8, pretrained_embedding=train_generator.umap_embedding_vectors, dropout_p=args.dropout_p)
        else:
            model = basemodel(n_labels=3, hidden_size=32, embedding_dim=8, dropout_p=args.dropout_p)
        # optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        optim = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)

        for epoch in range(1, args.n_epoch + 1):
            print(f'<<Epoch {epoch}>>', end='\n')
            train(model, train_loader, optim, epoch, verbose=1)
            model_metrics = test(model, test_loader)
            # save after last epoch
            model_metrics.update({'model':name, 'test_frac':args.test_frac, 'epoch':epoch})
            result_metrics.append(model_metrics)
            
    # # save results        
    #         if name in EMB_MODELS:
    #             emb_df = pd.DataFrame(index=train_generator.station_embeddings.sid, data=model.sid_embedding.weight.clone().detach().numpy(), columns=['dim_' + str(i) for i in range(8)])
    #             emb_df.to_csv(f'./embedding_tables/embdf_model-{name}_testfrac-{args.test_frac}_predstep-{args.pred_step}_epoch-{epoch}.csv', index=False)

    # result_df = pd.DataFrame(result_metrics)
    # result_df.to_csv(f'./metric_table/timewise_metrics_testfrac-{args.test_frac}_predstep-{args.pred_step}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for running model comparison')
    
    parser.add_argument('--test_frac', default='0.2', type=str)
    parser.add_argument('--n_in', default=12, type=int)
    parser.add_argument('--n_out', default=6, type=int)
    parser.add_argument('--n_hist', default=4, type=int)
    parser.add_argument('--history_smoothing', action='store_true')
    parser.add_argument('--pred_step', default=3, type=int)
    parser.add_argument('--dropout_p', default=0.2, type=float)
    parser.add_argument('--n_epoch', default=10, type=int)
    args = parser.parse_args()

    run(args)
