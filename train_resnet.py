import time
from comet_ml import Experiment
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from dataloader import dataloader
import copy

from train_model import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_loader, train_size, valid_loader, valid_size, test_loader, test_size = dataloader(batch_size=4, transform=transform)
dataloader = {'train': train_loader, 'val': valid_loader}

model = models.vgg16()
model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=9)
# print(model)

if use_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_fit = train_model(model, criterion, optimizer, num_epochs=100)
            
