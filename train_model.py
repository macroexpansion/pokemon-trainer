import time
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from dataloader import dataloader
import copy
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping


def train_model(model, loss, optimizer, dataloader, train_size, valid_size, model_name='weights', num_epochs=50):

    writer = SummaryWriter(comment='--{}'.format(model_name))
    es = EarlyStopping(patience=5)
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        start = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss_ = loss(outputs, labels)

                    if phase == 'train':
                        loss_.backward()
                        optimizer.step()
                
                running_loss += loss_.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            data_size = train_size if phase == 'train' else valid_size
            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects / data_size
            print(epoch_acc.dtype)

            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/test', epoch_loss, epoch)
                writer.add_scalar('Accuracy/test', epoch_acc, epoch)

            print('{} -> Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('\ttime', time.time() - start)

            if phase == 'val':
                acc = epoch_acc.cpu()
                if es.step(acc):
                    time_elapsed = time.time() - since
                    print('Early Stopping')
                    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                    print('Best val Acc: {:4f}'.format(best_acc))
                    return

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print('Update best acc: {:4f}'.format(best_acc))
                    torch.save(model.state_dict(), '{}.pt'.format(model_name))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))