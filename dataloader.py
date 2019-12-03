import torch
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # dataset = PokemonDataset('datasets/')
    image_data = ImageFolder(root='datasets/', transform=ToTensor())

    X, y = [], []
    for i in image_data:
        tensor, label = i
        X.append(tensor)
        y.append(label)

    print(len(X))

    # x_train, x_test, y_train, y_test = train_test_split()
    # print(data.classes) # list of classes
    # img, target = data[0]
    # plt.imshow(img.permute(1,2,0))
    # plt.show()