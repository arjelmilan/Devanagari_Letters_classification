import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import zipfile
import os
from steps import train
from model import Net


def extract_zip(zip_file_path, extract_to_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(extract_to_folder, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the destination folder
        zip_ref.extractall(extract_to_folder)


zip_file_path = 'archive (2).zip'
extract_to_folder = 'data'
extract_zip(zip_file_path, extract_to_folder)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = v2.Compose([
    ToTensor()
])

# Create dataset
train_dataset = ImageFolder("data/DEVNAGARI_NEW/TRAIN",
                            transform=transforms)
test_dataset = ImageFolder("data/DEVNAGARI_NEW/TEST",
                           transform=transforms)

# Create dataloader
train_dataloader = DataLoader(train_dataset,
                            batch_size = 32,
                            shuffle = True)
test_dataloader = DataLoader(test_dataset,
                            batch_size = 32,
                            shuffle = True)


model_2 = Net(input_shape=3,
              output_shape=48)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(),lr=0.001)
result =train(model=model_2,
      train_data = train_dataloader,
      test_data=test_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      epochs=8,
              device=device)
print(result)


