import random

import torch
import numpy as np

from src.models.cifar_resnet_large import ResNet

manual_seed = 0

np.random.seed(manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    model = ResNet((5, 3, 3), use_gpu=True)
    model.validation_accuracy(subset=False)

    # optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1, weight_decay=0)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150])
    lr_scheduler = None

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=1e-6)
    model.train_model(optimizer, lr_scheduler, epochs=50)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-6)
    model.train_model(optimizer, lr_scheduler, epochs=100)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=1e-6)
    model.train_model(optimizer, lr_scheduler, epochs=50)

    print(f"Final test accuracy: {model.validation_accuracy(subset=False):.2f}")

    model.save()
