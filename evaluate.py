import time
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
import nets
from dataloader import testloader


def evaluate(model, test_loader, model_name='weights.pt'):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('Evaluate Using CUDA')

    path = '../drive/My Drive/Colab Notebooks/' + model_name
    writer = SummaryWriter(comment='--{}--evaluate'.format(model_name))
    
    model.load_state_dict(torch.load(path))
    model.eval()

    if use_gpu:
      model.cuda()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
    print('Accuracy:', 100 * correct / total)
    writer.add_scalar('Accuracy', 100 * correct / total, 1)


if __name__ == '__main__':
    # net = nets.VGG16(pretrained=False)
    net = nets.VGG16_BN(pretrained=True)
    # net = nets.ResNet50(pretrained=False)
    # net = nets.MobileNetv2(pretrained=False)

    test_loader = testloader(colab=True)
    evaluate(net, test_loader, model_name='vgg16_bn_pretrained_augmented_96batch.pt')