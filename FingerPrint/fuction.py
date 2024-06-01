from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

class SIN(nn.Module):
    def __init__(self): 
        super(SIN, self).__init__() 
    def forward(self, x):
        return tc.sin(x)
    
class CustomImageDataset(Dataset):
    def __init__(self, root_dir,labels):
        self.root_dir   = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        image = image / image.max()  # Normaliza a imagem para o intervalo [0, 1]
        image = 1 - image
        image = tc.tensor(image, dtype=tc.float32)
        image = image.unsqueeze(0)  # Adiciona uma dimensão de canal se for uma imagem em escala de cinza
        
        label = tc.tensor(self.labels[idx].astype(int),dtype=tc.float32)
        return image,label

class FingerEncoder(nn.Module):
    def __init__(self, neck):
        super(FingerEncoder, self).__init__()
        # Encoder CNN
        self.conv1   = nn.Conv2d(1, 5, kernel_size=4, stride=1, padding=1) 
        self.conv2   = nn.Conv2d(5, 10, kernel_size=4, stride=1, padding=1)  
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) 
        self.encode3 = nn.Linear(10*40*40, 40*40)  # Encode
        self.encode2 = nn.Linear(40*40, 40)  # Encode
        self.encode1 = nn.Linear(40, neck)  # Encode
        self.act     = SIN()
        self.act2    = nn.Sigmoid()

    def forward(self, x):
        # Encoder CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Encoder Autoencoder
        x = x.view(-1, 10*40*40)  # Flatten before fully connected layers
        x = self.act(self.encode3(x))
        x = self.act(self.encode2(x))
        x = self.act2(self.encode1(x))   
        return x

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, batch_size=15, lr=0.001, step_size=500, gamma=0.9, device='cpu', pltrue=True):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = nn.BCEWithLogitsLoss()  # nn.L1Loss()#nn.MSELoss()
        self.optimizer = tc.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.losses = []
        self.dev = device
        self.pltrue = pltrue

    def train(self, epochs):
        self.model.to(self.dev)
        self.model.train()
        for epoch in tqdm(range(epochs)):
            for _, (data, label) in enumerate(self.train_dataloader, 0):
                data = data.to(self.dev)  # Move data to GPU
                label = label.to(self.dev)  # Move data to GPU
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                outputs = outputs.argmax(dim=1).view(-1,1).type(tc.float32)
                loss = self.criterion(outputs, label)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.losses.append(loss.item())
        print('Treinamento concluído')
        self.plot_losses(self.pltrue)

    def plot_losses(self, condi=True):
        if condi:
            plt.plot(self.losses)
            plt.yscale("log")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.show()
