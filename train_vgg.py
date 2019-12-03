import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from dataloader import dataloader
from train_model import train_model
from evaluate import evaluate


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

vgg = models.vgg16()
vgg.classifier[-1] = nn.Linear(in_features=4096, out_features=9)
# print(vgg)

if use_gpu:
    vgg.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.parameters(), lr=1e-4)

train_model(vgg, criterion, optimizer, dataloader, train_size, valid_size, model_name='vgg16', num_epochs=50)
            
