import torch 


def predict(model, test_loader, model_name='weights.pt'):
    use_gpu = torch.cuda.is_available()
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

    return preds_, labels_