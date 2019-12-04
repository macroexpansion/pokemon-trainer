import time
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from dataloader import testloader


def evaluate(model, model_name='weights.pt'):
    path = '../drive/My Drive/Colab Notebooks/' + model_name
    writer = SummaryWriter(comment='--{}--evaluate'.format(model_name))
    
    test_loader = testloader()
    model.load_state_dict(torch.load(path))
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labesl = labels.cuda()

            outputs = model(inputs)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy:', 100 * correct / total)
    writer.add_scalar('Accuracy', 100 * correct / total, 1)


if __name__ == '__main__':
    pass