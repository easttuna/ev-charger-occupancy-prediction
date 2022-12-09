import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import EvcDataset

from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score
from collections import Counter

class MultiSeqBase(nn.Module):
    def __init__(self, n_labels, hidden_size, embedding_dim, dropout_p):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=144, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(3*embedding_dim, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,n_labels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.gelu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.gelu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.gelu(self.fc_b1(fc_in))
        feature_vec = F.gelu(self.fc_b2(feature_vec))
        feature_vec = F.gelu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.gelu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return F.log_softmax(fc_out, dim=1)


class MultiSeqHybrid(nn.Module):
    def __init__(self, n_labels, hidden_size, embedding_dim, dropout_p):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=144, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding(num_embeddings=1_000, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(4*embedding_dim + 15, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,n_labels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.gelu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.gelu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec, s[:,1:]), dim=1)
        feature_vec = F.gelu(self.fc_b1(fc_in))
        feature_vec = F.gelu(self.fc_b2(feature_vec))
        feature_vec = F.gelu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.gelu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return F.log_softmax(fc_out, dim=1)


class MultiSeqUmap(nn.Module):
    def __init__(self, n_labels, hidden_size, embedding_dim, pretrained_embedding, dropout_p, freeze_embedding=True):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=144, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.fc_b1 = nn.Linear(3*embedding_dim + self.sid_embedding.weight.shape[1], 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,n_labels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.gelu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.gelu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec), dim=1)
        feature_vec = F.gelu(self.fc_b1(fc_in))
        feature_vec = F.gelu(self.fc_b2(feature_vec))
        feature_vec = F.gelu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.gelu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return F.log_softmax(fc_out, dim=1)


class MultiSeqHybridUmap(nn.Module):
    def __init__(self, n_labels, hidden_size, embedding_dim, pretrained_embedding, dropout_p, freeze_embedding=True):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=144, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.fc_b1 = nn.Linear(3*embedding_dim + self.sid_embedding.weight.shape[1] + 15, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,n_labels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.gelu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.gelu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec, s[:,1:]), dim=1)
        feature_vec = F.gelu(self.fc_b1(fc_in))
        feature_vec = F.gelu(self.fc_b2(feature_vec))
        feature_vec = F.gelu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.gelu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return F.log_softmax(fc_out, dim=1)



class MultiSeqUmapEarlyGating(nn.Module):
    def __init__(self, n_labels, hidden_size, embedding_dim, pretrained_embedding, dropout_p, freeze_embedding=True):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=144, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.fc_b1 = nn.Linear(3*embedding_dim + self.sid_embedding.weight.shape[1], 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.gating = nn.Linear(3*embedding_dim + self.sid_embedding.weight.shape[1],2)
        self.softmax = nn.Softmax(dim=1)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,n_labels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.gelu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.gelu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec), dim=1)
        feature_vec = F.gelu(self.fc_b1(fc_in))
        feature_vec = F.gelu(self.fc_b2(feature_vec))
        feature_vec = F.gelu(self.fc_b3(feature_vec))

        # apply gating
        weights = self.softmax(self.gating(fc_in))
        realtime_weight, history_weight = weights[:,:1], weights[:,1:]
        realtime_vec = realtime_vec * realtime_weight
        history_vec = history_vec * history_weight

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.gelu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return F.log_softmax(fc_out, dim=1)



class MultiSeqHybridUmapEarlyGating(nn.Module):
    def __init__(self, n_labels, hidden_size, embedding_dim, pretrained_embedding, dropout_p, freeze_embedding=True):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=144, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.fc_b1 = nn.Linear(3*embedding_dim + self.sid_embedding.weight.shape[1] + 15, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.gating = nn.Linear(3*embedding_dim + self.sid_embedding.weight.shape[1] + 15 ,2)
        self.softmax = nn.Softmax(dim=1)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,n_labels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.gelu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.gelu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec, s[:,1:]), dim=1)
        feature_vec = F.gelu(self.fc_b1(fc_in))
        feature_vec = F.gelu(self.fc_b2(feature_vec))
        feature_vec = F.gelu(self.fc_b3(feature_vec))

        # apply gating
        weights = self.softmax(self.gating(fc_in))
        realtime_weight, history_weight = weights[:,:1], weights[:,1:]
        realtime_vec = realtime_vec * realtime_weight
        history_vec = history_vec * history_weight

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.gelu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return F.log_softmax(fc_out, dim=1)


if __name__ == '__main__':

    # hidden_size, station_embedding_dim, embedding_dim
    HIDDEN_SIZE = 32
    STATION_EMBEDDING_DIM = 8
    EMBEDDING_DIM = 8

    R_seq = np.random.rand(16, 12, 1)
    H_seq = np.random.rand(16, 4, 1)
    T = np.hstack([
        np.random.choice(range(144), (16, 1)),
        np.random.choice(range(7), (16,1)),
        np.random.choice(range(2), (16,1))
        ])
    S = np.hstack([
        np.random.choice(range(20), (16,1)),
        np.random.rand(16, 15)
    ])
    Y_seq = np.random.choice(range(3), (16,1))

    print(R_seq.shape, H_seq.shape, Y_seq.shape, T.shape, S.shape)

    dataset = EvcDataset(R_seq, H_seq, T, S, Y_seq)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    r, h, t, s, y = next(iter(dataloader))

    # model = MultiSeqHybrid(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM)
    # model = MultiSeqUmap(n_labels=3, hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, dropout_p=0.2, pretrained_embedding=torch.tensor(np.random.rand(20, 8)).float())
    # model = MultiSeqHybridUmap(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, pretrained_embedding=torch.tensor(np.random.rand(20, 8)).float())
    # model = MultiSeqUmapGating(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, pretrained_embedding=torch.tensor(np.random.rand(20, 8)).float())
    model = MultiSeqHybridUmapEarlyGating(n_labels=3, hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, dropout_p=0.2, pretrained_embedding=torch.tensor(np.random.rand(20, 8)).float())

    pred = model(r, h, t, s)
    pred_labels = pred.argmax(dim=1, keepdim=True)

    loss = F.nll_loss(pred, y.flatten())
    print(loss.item())
    print(f1_score(y, pred_labels, average='macro'))
    print(f1_score(y, pred_labels, average='micro'))
    print(f1_score(y, pred_labels, average='weighted'))
    print('recall {}'.format(recall_score(y, pred_labels, labels=[0, 1, 2], average=None)))

    print(y.flatten())
    print(torch.exp(pred))
    auc_macro = roc_auc_score(y.flatten(), torch.exp(pred).detach().numpy(), average='macro', multi_class='ovr')
    auc_weighted = roc_auc_score(y.flatten(), torch.exp(pred).detach().numpy(), average='weighted', multi_class='ovr')
    print(auc_macro, auc_weighted)