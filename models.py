import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import EvcDataset

class NaiveModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r, h, t, s):
        return r[:,-1,:]


class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r, h, t, s):
        return h.mean(axis=1)


class BaseHybrid(nn.Module):
    def __init__(self, hidden_size, station_embedding_dim, embedding_dim):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.station_embedding = nn.Embedding(num_embeddings=2000, embedding_dim=station_embedding_dim)
        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(station_embedding_dim+ 3*embedding_dim, 128)
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
        station_vec = self.station_embedding(s[:,0])
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((station_vec, timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class GatingHybrid(nn.Module):
    def __init__(self, hidden_size, station_embedding_dim, embedding_dim):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.station_embedding = nn.Embedding(num_embeddings=2000, embedding_dim=station_embedding_dim)
        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(station_embedding_dim+ 3*embedding_dim, 128)
        self.fc_b2 = nn.Linear(128, 64)
        self.fc_b3 = nn.Linear(64, 64)

        self.gating = nn.Linear(64,2)
        self.softmax = nn.Softmax(dim=1)

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
        station_vec = self.station_embedding(s[:,0])
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((station_vec, timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # gating function
        weights = self.softmax(self.gating(feature_vec))
        realtime_weight, history_weight = weights[:,:1], weights[:,1:]
        realtime_vec = realtime_vec * realtime_weight
        history_vec = history_vec * history_weight

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)

        
class DcodeHybrid(nn.Module):
    def __init__(self, hidden_size, station_embedding_dim, embedding_dim):
        super().__init__()
        self.lstm_r1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_r1 = nn.Linear(hidden_size, 16)
    
        self.lstm_h1 = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc_h1 = nn.Linear(hidden_size, 16)

        self.station_embedding = nn.Embedding(num_embeddings=2000, embedding_dim=station_embedding_dim)
        self.dcode_embedding = nn.Embedding(num_embeddings=20, embedding_dim=embedding_dim)
        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.fc_b1 = nn.Linear(station_embedding_dim+ 4*embedding_dim, 128)
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
        station_vec = self.station_embedding(s[:,0])
        dcode_vec = self.dcode_embedding(s[:,1])
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])

        fc_in = torch.cat((station_vec, dcode_vec, timeslot_vec, dow_vec, we_vec), dim=1)
        feature_vec = F.relu(self.fc_b1(fc_in))
        feature_vec = F.relu(self.fc_b2(feature_vec))
        feature_vec = F.relu(self.fc_b3(feature_vec))

        # concatenation
        cat_vec = torch.cat((realtime_vec, history_vec, feature_vec), dim=1)
        fc_out = F.relu(self.fc_cat(cat_vec))
        fc_out = self.dropout(fc_out)
        fc_out = self.top(fc_out)
        return torch.sigmoid(fc_out)


class Fastfusion(nn.Module):
    def __init__(self, hidden_size, station_embedding_dim, embedding_dim):
        super().__init__()

        self.station_embedding = nn.Embedding(num_embeddings=2000, embedding_dim=station_embedding_dim)
        self.timeslot_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embedding_dim)
        self.dow_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.we_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(station_embedding_dim+3*embedding_dim+1, hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,1)

    def forward(self, r, h, t, s):
        # non-sequenctials
        station_vec = self.station_embedding(s[:,0])
        timeslot_vec = self.timeslot_embedding(t[:,0])
        dow_vec = self.dow_embedding(t[:,1])
        we_vec = self.we_embedding(t[:,2])
        embedding_vecs = torch.cat((station_vec, timeslot_vec, dow_vec, we_vec), dim=1)
        embedding_vecs = torch.unsqueeze(embedding_vecs, 1)
        embedding_vecs = embedding_vecs.repeat((1,12,1))

        lstm_in = torch.cat((embedding_vecs, r), dim=2)
        lstm_out, (hn, cn) = self.lstm(lstm_in)
        last_state = lstm_out[:,-1,:]
        last_state = F.relu(self.dropout(last_state))

        fc_out = F.relu(self.fc1(last_state))
        fc_out = F.relu(self.fc2(fc_out))
        fc_out = self.fc3(fc_out)
        
        return torch.sigmoid(fc_out)


if __name__ == '__main__':

    # hidden_size, station_embedding_dim, embedding_dim
    HIDDEN_SIZE = 8
    STATION_EMBEDDING_DIM = 8
    EMBEDDING_DIM = 8

    R = np.random.rand(16, 12, 1)
    H = np.random.rand(16, 3, 1)
    T = np.hstack([
        np.random.choice(range(48), (16, 1)),
        np.random.choice(range(7), (16,1)),
        np.random.choice(range(2), (16,1))
        ])
    S = np.random.choice(range(2000), (16, 2))
    Y = np.random.rand(16,1)

    dataset = EvcDataset(R, H, T, S, Y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    r, h, t, s, y = next(iter(dataloader))

    model = GatingHybrid(HIDDEN_SIZE, STATION_EMBEDDING_DIM, EMBEDDING_DIM)
    pred = model(r, h, t, s)
    print(pred.shape)
