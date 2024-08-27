import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import os


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

def parse_dict(dict_str):
    try:
        parsed = json.loads(dict_str)
        data_array = [np.interp(np.linspace(0, len(parsed_value) - 1, 280), np.arange(len(parsed_value)), parsed_value) for parsed_value in parsed.values()]
        raw = np.vstack(data_array)
        return raw
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return None

##

class TraceDataset(Dataset):
    def __init__(self, df):
        self.data = df['data']
        self.labels = df['channel']

    @classmethod
    def fromCsv(cls, csv_path: str, cache_pickle=True):
        pickle_path = csv_path + '.pkl'
        pickle_exists = os.path.isfile(pickle_path)

        csv_mod_time = os.path.getmtime(csv_path)
        pickle_mod_time = os.path.getmtime(pickle_path) if pickle_exists else 0

        if cache_pickle:
            if not pickle_exists or csv_mod_time > pickle_mod_time:
                df = pd.read_csv(csv_path, dtype=dtype_dict)
                df['data'] = df['data'].apply(parse_dict)
                df.to_pickle(pickle_path)
            else:
                df = pd.read_pickle(pickle_path)
        else:
            df = pd.read_csv(csv_path, dtype=dtype_dict)
            df['data'] = df['data'].apply(parse_dict)
        return cls(df)
    
    @classmethod
    def fromPickle(cls, pickle_path: str):
        df = pd.read_pickle(pickle_path)
        return cls(df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x = self.data.iloc[idx][:,:200]
        x = self.data.iloc[idx]
        y = self.labels.iloc[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    # trace_dataset = TraceDataset.fromCsv("assets/csv/ThailandV2.csv")
    trace_dataset = TraceDataset.fromPickle("assets/csv/ThailandV2.csv.pkl")
    trace_dataset.data
