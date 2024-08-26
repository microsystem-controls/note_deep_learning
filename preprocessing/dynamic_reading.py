import pandas as pd
import numpy as np
import ast

dtype_dict = {
    'path': str,
    'country' : str,
    'isGenuine' : bool, #bool,
    'channel' : int,  #int,
    'revision' : str,
    'orientation' : str,
    'validatorModel' : str,
    'serial' : str,
    'firmwareRevision' : str,
    'created' : str,
    'insertion' : int, #int
    'sensor' : str,
    'data' : str
    }

# df = pd.read_csv("assets/csv/Thailand.csv", dtype=dtype_dict)
df = pd.read_csv("assets/csv/test.csv", dtype=dtype_dict)

def parse_dict(dict_str):
    try:
        parsed = ast.literal_eval(dict_str)
        return np.vstack(list(parsed.values()))
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return None

df['data'] = df['data'].apply(parse_dict)

##

import torch
from torch.utils.data import Dataset

class TraceDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe['data']
        self.labels = dataframe['channel']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x = self.data.iloc[idx][:,:200]
        x = self.data.iloc[idx]
        y = self.labels.iloc[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

##

from torch.utils.data import DataLoader

# Create the dataset
dataset = TraceDataset(df)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

data_iter = iter(dataloader)
i = next(data_iter)
df['data'].iloc[0]
