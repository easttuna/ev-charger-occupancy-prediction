import torch
from torch.utils.data import Dataset


class EvcDataset(Dataset):
    def __init__(self, xs, sids, ys):
        assert len(xs) == len(ys)

        self.xs = torch.tensor(xs).float()
        self.sids = torch.tensor(sids)
        self.ys = torch.tensor(ys).float()
        
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x, sid, y = self.xs[i], self.sids[i], self.ys[i]
        return x, sid, y

