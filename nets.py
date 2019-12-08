import torch.nn as nn
from torchvision import models


def VGG16(pretrained=False):
    net = models.vgg16(pretrained=pretrained)
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=9)

    return net


def VGG16_BN(pretrained=False):
    net = models.vgg16_bn(pretrained=pretrained)
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=9)

    return net


def ResNet50(pretrained=False):
    net = models.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(in_features=2048, out_features=9)

    return net


def MobileNetV2(pretrained=False):
    net = models.mobilenet_v2(pretrained=pretrained)

    for param in net.parameters():
        param.requires_grad = False

    net.classifier = nn.Sequential(nn.Dropout(0.3), 
                                   nn.Linear(1280, 640), 
                                   nn.Dropout(0.3),
                                   nn.Linear(640, 9))

    return net


def MobileNetV2_Normal(pretrained=False):
    net = models.mobilenet_v2(pretrained=pretrained)

    for param in net.parameters():
        param.requires_grad = False

    net.classifier[-1] = nn.Linear(in_features=1280, out_features=9) 

    return net


if __name__ == '__main__':
    net = MobileNetV2()
    print(net)