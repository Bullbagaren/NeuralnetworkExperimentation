import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd



class ParquetDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path, columns=["token", "tag"])

        self.token2idx = {t: i+1 for i, t in enumerate(df["token"].unique())}
        self.tag2idx = {t:i for i, t in enumerate(df["tag"].unique())}
        self.idx2tag = {v: k for k,v in self.tag2idx.items()}

        self.tokens = df["token"].map(self.token2idx).values
        self.tags = df["tag"].map(self.tag2idx).values

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index], dtype=torch.long)
        y = torch.tensor(self.tags[index], dtype=torch.long)
        return x, y



