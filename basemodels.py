import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import EvcDataset


class HistoricBase(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super().__init__()
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        # for time-dependent features
        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(3*embedding_dim, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(16+64, 64)
        self.top = nn.Linear(64,1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, r, h, t, s):
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.relu(self.fc_h1(history_vec))

        # non-sequenctials - time related features
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class RealtimeBase(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(3*embedding_dim, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(16+64, 64)
        self.top = nn.Linear(64,1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.relu(self.fc_r1(realtime_vec))

        # non-sequenctials - time related features
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class MultiSeqBase(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(3*embedding_dim, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.relu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.relu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class MultiSeqHybrid(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding(num_embeddings=1_000, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(4*embedding_dim + 15, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.relu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.relu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec, s[:,1:]), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class MultiSeqUmap(nn.Module):
    def __init__(self, hidden_size, embedding_dim, pretrained_embedding):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)

        self.fc_b1 = nn.Linear(4*embedding_dim + 15, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.relu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.relu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec, s[:,1:]), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class MultiSeqUmapEmbonly(nn.Module):
    def __init__(self, hidden_size, embedding_dim, pretrained_embedding):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        self.sid_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)

        self.fc_b1 = nn.Linear(4*embedding_dim, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.fc_cat = nn.Linear(32+64, 64)
        self.top = nn.Linear(64,1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, r, h, t, s):
        # realtime sequence
        lstm_out_r, (hn, cn) = self.lstm_r1(r)
        last_state_r = lstm_out_r[:,-1,:]
        realtime_vec = self.dropout(last_state_r)
        realtime_vec = F.relu(self.fc_r1(realtime_vec))
        
        # history sequence
        lstm_out_h, (hn, cn) = self.lstm_h1(h)
        last_state_h = lstm_out_h[:,-1,:]
        history_vec = self.dropout(last_state_h)
        history_vec = F.relu(self.fc_h1(history_vec))

        # non-sequenctials
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        sid_vec = self.sid_embedding(s[:,0].int())

        fc_in = torch.cat((timeslot_vec, dow_vec, we_vec, sid_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


if __name__ == '__main__':

    # hidden_size, station_embedding_dim, embedding_dim
    HIDDEN_SIZE = 8
    STATION_EMBEDDING_DIM = 8
    EMBEDDING_DIM = 8

    R_seq = np.random.rand(16, 12, 1)
    H_seq = np.random.rand(16, 4, 1)
    T = np.hstack([
        np.random.choice(range(48), (16, 1)),
        np.random.choice(range(7), (16,1)),
        np.random.choice(range(2), (16,1))
        ])
    S = np.hstack([
        np.random.choice(range(20), (16,1)),
        np.random.rand(16, 15)
    ])
    Y_seq = np.random.rand(16,1)

    print(R_seq.shape, H_seq.shape, Y_seq.shape, T.shape, S.shape)

    dataset = EvcDataset(R_seq, H_seq, T, S, Y_seq)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    r, h, t, s, y = next(iter(dataloader))

    # model = MultiSeqHybrid(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM)
    model = MultiSeqUmap(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, pretrained_embedding=torch.tensor(np.random.rand(20, 8)).float())
    model = MultiSeqUmapEmbonly(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, pretrained_embedding=torch.tensor(np.random.rand(20, 8)).float())

    pred = model(r, h, t, s)
    print(pred.shape)
