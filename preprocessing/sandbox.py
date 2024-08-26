from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocessing.dataset import TracesDataset
import torch
import os
import pandas as pd
from torchvision.io import read_image

## example of loading Thailand dataset and dataloader
country = "Thailand"
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

figure = plt.figure(figsize=(8, 8))
cols, rows = 7, 7
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(train_data.index_to_label[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), aspect="auto")
plt.show()


##
image = read_image(os.path.join(img_dir, '89THAILANDGenuineTH0050CTH0050DLUST217-23-01981671404109201765823AM(60notes)_0.jpg'))
image = image.squeeze()

row_image = image[0, :]
plt.plot(row_image.numpy())
plt.show(block=False)


df = pd.read_csv("assets/csv/Thailand.csv", nrows=60)

row = df[df["sensor"] == "MiddleLeftIr"] # this happens to be the first one

plt.figure()
path_to_image = row["path"].tolist()[0]
print(f"path_to_image = {path_to_image}")
plt.plot(list(map(int, row["data"].tolist()[0].split(" "))))
plt.show()


##

