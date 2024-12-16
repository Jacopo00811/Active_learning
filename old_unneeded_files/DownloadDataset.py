import torchvision
from torchvision.transforms import v2 as transforms
import torch

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),     # subtract 0.5 and divide by 0.5
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
