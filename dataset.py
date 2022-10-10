import torch
from torch.utils.data import Dataset


class EvcBaseDataset(Dataset):
    def __init__(self, xs, ys):
        assert len(xs) == len(ys)

        self.xs = torch.tensor(xs).float()
        self.ys = torch.tensor(ys).float()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x, y = self.xs[i], self.ys[i]
        return x, y


class EvcEmbeddingDataset(EvcBaseDataset):
    def __init__(self, xs, es, ys):
        assert len(xs) == len(ys)

        self.xs = torch.tensor(xs).float()
        self.es = torch.tensor(es)
        self.ys = torch.tensor(ys).float()
        
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x, e, y = self.xs[i], self.es[i], self.ys[i]
        return x, e, y
