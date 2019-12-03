import torch
import numpy as np

def calc_norm(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1147, shuffle=False, num_workers=4)
    pop_mean = []
    pop_std0 = []
    pop_std1 = []

    for i, data in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data['image'].numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    
    print(pop_mean)
    print(pop_std0)
    print(pop_std1)
