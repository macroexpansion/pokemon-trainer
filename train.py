import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms
import nets
from dataloader import dataloader
from train_model import train_model
from evaluate import evaluate
from utils import ImgAugmenter
import PIL

normalize = transforms.Normalize(mean=[0.6855248, 0.68901044, 0.6142709], std=[0.32218322, 0.27970782, 0.3134101])

preprocess = {
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

augmented = {
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
                                                                transform=preprocess)
dataloader = {'train': train_loader, 'val': valid_loader}


if __name__ == '__main__':
    # net = nets.VGG16(pretrained=False)
    # net = nets.VGG16_BN(pretrained=False)
    # net = nets.ResNet50(pretrained=False)
    net = nets.MobileNetV2(pretrained=False)
    # net = nets.MobileNetV2_Normal(pretrained=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)

    train_model(net, criterion, optimizer, dataloader, train_size, valid_size, 
                model_name='mobilenetv2_augment', 
                num_epochs=100)