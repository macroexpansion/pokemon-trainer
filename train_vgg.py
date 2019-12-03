import time
from comet_ml import Experiment
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from dataloader import dataloader
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_loader, train_size, valid_loader, valid_size, test_loader, test_size = dataloader(batch_size=4, transform=transform)
dataloader = {'train': train_loader, 'val': valid_loader}

model = models.vgg16()
model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=9)
# print(model)

if use_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, criterion, optimizer, scheduler=None, num_epochs=100):
    experiment = Experiment(api_key='9tplk9L0Vy6WW3K7rmdbEz7jm', project_name='pkm-trainer')
    experiment.add_tags(['vgg16', 'no_batchnorm', 'no_pretrained'])
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with experiment.train():
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() #* inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    del inputs, labels, outputs, preds
                    torch.cuda.empty_cache()

                data_size = train_size if phase == 'train' else valid_size
                epoch_loss = running_loss / data_size
                epoch_acc = running_corrects.double() / data_size
                if phase == 'train':
                    experiment.log_metric('train_loss', epoch_loss, step=epoch)
                    experiment.log_metric('train_acc', epoch_acc, step=epoch)
                else:
                    experiment.log_metric('val_loss', epoch_loss, step=epoch)
                    experiment.log_metric('val_acc', epoch_acc, step=epoch)

                print('{} -> Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model_fit = train_model(model, criterion, optimizer, num_epochs=100)
            
