import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from dataloader import dataloader
from train_model import train_model
from evaluate import evaluate


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

normalize = transforms.Normalize(mean=[0.6855248, 0.68901044, 0.6142709], std=[0.32218322, 0.27970782, 0.3134101])
transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
}

train_loader, train_size, valid_loader, valid_size, test_loader, test_size = dataloader(colab=True, 
                                                                                        batch_size=32, 
                                                                                        transform=transform)
dataloader = {'train': train_loader, 'val': valid_loader}

mobilenet = models.mobilenet_v2()
mobilenet.classifier[-1] = nn.Linear(in_features=1280, out_features=9)
# print(mobilenet)

if use_gpu:
    resnet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=1e-4)

train_model(resnet, criterion, optimizer, dataloader, train_size, valid_size, model_name='mobilenetv2', num_epochs=100)
