import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from dataloader import dataloader
from train_model import train_model
from evaluate import evaluate
from utils import ImgAugmenter
import PIL

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

normalize = transforms.Normalize(mean=[0.6855248, 0.68901044, 0.6142709], std=[0.32218322, 0.27970782, 0.3134101])
transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ImgAugmenter(),
        lambda x: PIL.Image.fromarray(x),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15),
        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
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

train_loader, train_size, valid_loader, valid_size = dataloader(colab=True, 
                                                    batch_size=64, 
                                                    transform=transform)
dataloader = {'train': train_loader, 'val': valid_loader}

mobilenet = models.mobilenet_v2(pretrained=True)
for param in mobilenet.parameters():
    param.requires_grad = False
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.3), 
    nn.Linear(1280, 640), 
    nn.Dropout(0.3),
    nn.Linear(640, 9)
)
# print(mobilenet)

if use_gpu:
    mobilenet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet.parameters(), lr=1e-4)

# train_model(mobilenet, criterion, optimizer, dataloader, train_size, valid_size, model_name='mobilenetv2_augment', num_epochs=100)
evaluate(mobilenet, test_loader, model_name='mobilenetv2_augment.pt')