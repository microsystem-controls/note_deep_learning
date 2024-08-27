from preprocessing.dataset import TraceDataset
from torch.utils.data import DataLoader

# Create the dataset
# csv_file = "assets/csv/test.csv"
csv_file = "assets/csv/ThailandV2.csv"
dataset = TraceDataset(csv_file)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

data_iter = iter(dataloader)
i = next(data_iter)
i[0].shape


def parse_dict(dict_str):
    try:
        parsed = json.loads(dict_str)
        data_array = [np.interp(np.linspace(0, len(parsed_value) - 1, 280), np.arange(len(parsed_value)), parsed_value) for parsed_value in parsed.values()]
        raw = np.vstack(data_array)
        return raw
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return None
import pandas as pd

df = pd.read_csv(csv_file, dtype=dtype_dict)
df['data'] = df['data'].apply(parse_dict)
store = pd.HDFStore('store.h5')
df['data'].apply(lambda x: np.floor(x))
df['data']
store['df'] = df['data']
df['data']
df.to_pickle('store.pkl')

df = pd.read_pickle('store.pkl')
