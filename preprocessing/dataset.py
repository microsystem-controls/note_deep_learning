import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
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

def toOneHot(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class TraceDataset(Dataset):
    def __init__(self, df):
        self.data = df['data']
        self.orientation = df["orientation"].unique()
        self.channel = df["channel"].unique()

        # Create mappings from unique values to indices
        self.orientation_to_index = {value: index for index, value in enumerate(self.orientation)}
        self.channel_to_index = {value: index for index, value in enumerate(self.channel)}
        
        # Map the labels to numeric values
        df['labels_orientation'] = df['orientation'].map(self.orientation_to_index)
        df['labels_channel'] = df['channel'].map(self.channel_to_index)
        self.labels = df[["labels_orientation", "labels_channel"]]
        ## put labels orientation and labels channel into a df with the same headers as before

    def fromOneHot(self, y):
        def _fromOneHot(one_hot, index_to_value):
            index = one_hot.argmax(axis=1)
            return index_to_value[index.numpy()].tolist()
        if len(y.shape) == 1:
            y = y.expand(1, -1)
        y_orientation = _fromOneHot(y[:, 0:len(self.orientation)], self.orientation)
        y_channel = _fromOneHot(y[:, 0:len(self.channel)], self.channel)
        return y_orientation, y_channel
    
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
        len(self.orientation_to_index)
        y_orientation = toOneHot(y["labels_orientation"], len(self.orientation_to_index))
        y_channel = toOneHot(y["labels_channel"], len(self.channel_to_index))
        y = np.concatenate([y_orientation, y_channel], axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

##

if __name__ == "__main__":
    # trace_dataset = TraceDataset.fromCsv("assets/csv/ThailandV2.csv")
    # trace_dataset = TraceDataset.fromPickle("assets/csv/ThailandV2.csv.pkl")
    # trace_dataset.data
    trace_dataset = TraceDataset.fromCsv("assets/csv/test.csv")
    dataloader = DataLoader(trace_dataset, batch_size=32, shuffle=True)
    i = iter(dataloader)
    data = next(i)
    trace_dataset.fromOneHot(data[1])
