
from torchvision.models import resnet18, ResNet18_Weights
import torch

import ssl  # hack for remote ssl
ssl._create_default_https_context = ssl._create_unverified_context


model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)  # forward pass

loss = (prediction - labels).sum()
loss.backward()  # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step()  #gradient descent
