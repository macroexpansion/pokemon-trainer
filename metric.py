import time
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import confusion_matrix
from predict import predict


def metric(predicts, labels):
    pass