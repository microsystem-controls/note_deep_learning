import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class TracesDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.index_to_label = sorted(self.img_labels["channel"].unique())
        self.label_to_index = {value: i for i, value in enumerate(self.index_to_label)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.label_to_index[self.img_labels.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


