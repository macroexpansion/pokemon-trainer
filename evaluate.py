import torch
import nets
from dataloader import testloader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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


def get_prediction(model, test_loader, model_name='weights.pt'):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('Using CUDA')
    device = 'cuda:0' if use_gpu else 'cpu'

    path = '../drive/My Drive/Colab Notebooks/' + model_name
    
    model.load_state_dict(torch.load(path))
    model.eval()

    if use_gpu:
      model.cuda()

    preds_ = torch.tensor([], dtype=torch.int64, device=device)
    labels_ = torch.tensor([], dtype=torch.int64, device=device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            preds_ = torch.cat((preds_, predicted), dim=0)
            labels_ = torch.cat((labels_, labels), dim=0)

    preds_ = preds_.cpu()
    labels_ = labels_.cpu()
    return preds_, labels_


if __name__ == '__main__':
    # net = nets.VGG16(pretrained=False)
    # net = nets.VGG16_BN(pretrained=True)
    # net = nets.ResNet50(pretrained=False)
    # net = nets.MobileNetV2(pretrained=False)
    net = nets.MobileNetV2_Normal(pretrained=False)
    
    test_loader = testloader(colab=True)
    # evaluate(net, test_loader, model_name='vgg16_bn_pretrained_augmented_96batch.pt')
    pred, truth = get_prediction(net, test_loader, model_name='vgg16_bn_pretrained_augmented_96batch.pt')
    
    target_names = ['bulbasaur', 'charmander', 'jigglypuff', 'magikarp', 'mudkip', 'pikachu', 'psyduck', 'snorlax', 'squirtle']
    report = classification_report(truth, pred, target_names=target_names)
    print(report)
    matrix = confusion_matrix(truth, pred)
    df = pd.DataFrame(matrix, index = [i for i in target_names], columns = [i for i in target_names])
    fig = plt.figure(figsize = (10,7))
    sns.heatmap(df, annot=True, cbar=False, cmap="YlGnBu")
    plt.savefig('matrix')
    