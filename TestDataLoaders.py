from torch.utils.data import DataLoader
import torchvision
import torch
from torchvision.transforms import v2 as transforms


hyperparameters = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'num_workers': 0,
    'num_classes': 10,
    'input_size': 32 * 32 * 3,
    'hidden_size': 512,
    'output_size': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),     # subtract 0.5 and divide by 0.5
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=hyperparameters['batch_size'],
                          shuffle=True, num_workers=hyperparameters['num_workers'], drop_last=True)
test_loader = DataLoader(test_set, batch_size=hyperparameters['batch_size'],
                         shuffle=False, num_workers=hyperparameters['num_workers'], drop_last=False)


# Test the data loaders
classes = {index: name for name, index in train_set.class_to_idx.items()}
print("Classes:", classes)

print("\nTraining data")
print("Number of points:", len(train_set))
x, y = next(iter(train_loader))
print("Batch dimension (B x C x H x W):", x.shape)
print(f"Number of distinct labels: {
      len(set(train_set.targets))} (unique labels: {set(train_set.targets)})")

print("\nTest data")
print("Number of points:", len(test_set))
x, y = next(iter(test_loader))
print("Batch dimension (B x C x H x W):", x.shape)
print(f"Number of distinct labels: {
      len(set(test_set.targets))} (unique labels: {set(test_set.targets)})")

n_classes = len(set(test_set.targets))