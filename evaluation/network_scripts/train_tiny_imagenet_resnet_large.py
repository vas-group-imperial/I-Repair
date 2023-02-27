import random

import torch
import numpy as np

from src.models.tiny_imagenet_resnet_large import ResNet

manual_seed = 0

np.random.seed(manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# noinspection PyUnresolvedReferences
torch.cuda.manual_seed(manual_seed)
# noinspection PyUnresolvedReferences
torch.cuda.manual_seed_all(manual_seed)
# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    model = ResNet((5, 3, 3), use_gpu=True)
    model.validation_accuracy(subset=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train_model(optimizer, epochs=50)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train_model(optimizer, epochs=100)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=1e-4)
    model.train_model(optimizer, epochs=50)

    print(f"Final test accuracy: {model.validation_accuracy(subset=False):.2f}")

    model.save()
