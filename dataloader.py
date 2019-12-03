import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def datacounter(root='pkm/train/'):
    classes = ['bulbasaur', 'charmander', 'jigglypuff', 'magikarp', 'mudkip', 'pikachu', 'psyduck', 'snorlax', 'squirtle']
    X, y = [], []
    for index, clsname in enumerate(classes):
        imgpath = os.path.join(root, clsname)
        for path in os.listdir(imgpath):
            imgfile = os.path.join(imgpath, path)
            X.append(imgfile)
            y.append(index)
    return X, y


def dataplot(y_train, y_test):    
    f, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1, ax2 = axes
    sns.countplot(y_train, ax=ax1)
    ax1.set_xlabel('train')
    ax1.set_ylabel('number')
    ax1.set_ylim([0, 200])
    sns.countplot(y_test, ax=ax2)
    ax2.set_xlabel('test')
    ax2.set_ylabel('number')
    ax2.set_ylim([0, 200])
    plt.show()


def train_valid_split(dataset, valid_split_size=0.2, shuffle=True, random_seed=7):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split_size * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, valid_indices = indices[split:], indices[:split]
    # print(train_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    return train_sampler, len(train_indices), valid_sampler, len(valid_indices)


def dataloader(colab=True, batch_size=16, transform=ToTensor()):
    path = 'pkm/'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/pkm/'
    train_data = ImageFolder(root=path + 'train/', transform=transform)
    train_sampler, train_size, valid_sampler, valid_size = train_valid_split(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    test_data = ImageFolder(root=path + 'test/', transform=transform)
    # print(test_data.targets)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, train_size, valid_loader, valid_size, test_loader, len(test_data)

if __name__ == '__main__':
    # _, y_train = dataloader('pkm/train/')
    # _, y_test = dataloader('pkm/test/')
    # dataplot(y_train, y_test)

    pass