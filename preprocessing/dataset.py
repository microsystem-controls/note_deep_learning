import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt


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


country = "Australia"
annotations_file = f"assets/torch/{country}/labels.txt"
img_dir = f"assets/torch/{country}/imgs/"
train_data = TracesDataset(annotations_file, img_dir)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, aspect="auto")
plt.show()
print(f"Label: {label}")
##

import os
from PIL import Image

# Specify the directory containing the images
img_dir = r"assets/torch/Thailand/imgs/"

width_height = set()


i = 0
# Loop through each file in the directory
for img_name in os.listdir(img_dir):
    # Get the full path to the image file
    img_path = os.path.join(img_dir, img_name)
    
    # Open the image file
    with Image.open(img_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        # Print the image name and its dimensions
        width_height.add((width, height))
        if len(width_height) == 2:
            img.show()
            print(f"img_path = {img_path}")
            break
    i += 1
    if (i % 1000 == 0):
        print(f"width_height = {width_height}")


width_height
