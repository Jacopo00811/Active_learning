import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
INITIAL_SUBSET_RATIO = 0.1
TRAIN_SUBSET_RATIO = 0.8
BUDGET_BATCH_RATIO = 0.1
EPOCHS = 5

# Set random seed for reproducibility
torch.manual_seed(0)

# Setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(device)
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Load CIFAR-10 with transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 Dataset (Training and Test splits)
train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split into training and validation sets
train_size = int(TRAIN_SUBSET_RATIO * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Initial labeled and unlabeled splits
initial_subset_size = int(INITIAL_SUBSET_RATIO * train_size)
initial_train_indices = np.random.choice(len(train_dataset), initial_subset_size, replace=False)
labeled_dataset = Subset(train_dataset, initial_train_indices)
unlabeled_indices = list(set(range(len(train_dataset))) - set(initial_train_indices))
unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

# Data loaders
train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_accuracy = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Accuracy: {val_accuracy:.2f}%")

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)

# Uncertainty Sampling (Least Confidence)
def uncertainty_sampling_least_confidence(model, unlabeled_loader, n_samples):
    model.eval()
    uncertainties = []
    samples_indices = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(unlabeled_loader):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            max_confidences, _ = torch.max(softmax_outputs, dim=1)
            uncertainties.extend(1 - max_confidences.cpu().numpy())
            samples_indices.extend([idx] * images.size(0))

    uncertain_indices = np.argsort(uncertainties)[-n_samples:]
    selected_indices = [samples_indices[i] for i in uncertain_indices]
    return selected_indices

# Random Sampling Strategy
def random_sampling(unlabeled_loader, n_samples):
    indices = list(range(len(unlabeled_loader.dataset)))
    return np.random.choice(indices, n_samples, replace=False).tolist()

# Active Learning Loop with Query Strategy Selection and Model Reset
def active_learning_loop(model, train_loader, test_loader, unlabeled_dataset, strategy="uncertainty"):

    # Initialize labeled and unlabeled datasets
    total_train_size = len(train_dataset)
    initial_subset_size = min(int(INITIAL_SUBSET_RATIO * total_train_size), total_train_size - 1)
    labeled_indices = np.random.choice(total_train_size, initial_subset_size, replace=False)
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_indices = list(set(range(total_train_size)) - set(labeled_indices))
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

    train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    accuracies = []
    training_set_sizes = []

    for i in range(int(1 / BUDGET_BATCH_RATIO)):
        # Reset model weights at the start of each iteration
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        train_model(model, train_loader, test_loader, epochs=EPOCHS)
        test_accuracy = evaluate_model(model, test_loader)
        accuracies.append(test_accuracy)
        training_set_sizes.append(len(labeled_dataset))
        print(f"Iteration {i+1}, Test Accuracy: {test_accuracy:.2f}%")

        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)
        n_samples_to_label = int(BUDGET_BATCH_RATIO * len(train_dataset))

        if strategy == "uncertainty":
            selected_indices = uncertainty_sampling_least_confidence(model, unlabeled_loader, n_samples_to_label)
        elif strategy == "random":
            selected_indices = random_sampling(unlabeled_loader, n_samples_to_label)
        else:
            raise ValueError("Unsupported strategy! Use 'uncertainty' or 'random'.")

        labeled_indices = np.concatenate([labeled_indices, selected_indices])
        labeled_dataset = Subset(train_dataset, labeled_indices)
        unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))
        unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
        train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)

    plt.figure()
    plt.plot(training_set_sizes, accuracies, marker='o')
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Active Learning: Accuracy vs Training Set Size ({strategy.capitalize()} Sampling)')
    plt.grid(True)
    plt.show()

# Full Training Data Loader (using all labeled data)
full_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Function to train model on full data for baseline performance
def baseline_full_data_performance(model, train_loader, test_loader, epochs=EPOCHS):
    model.apply(lambda m: type(m).reset_parameters(m) if hasattr(m, 'reset_parameters') else None)
    print("Training on the full dataset to find maximum performance baseline...")
    train_model(model, train_loader, test_loader, epochs=epochs)
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Maximum Performance Baseline - Test Accuracy: {test_accuracy:.2f}%")

# Run the baseline performance training
baseline_full_data_performance(model, full_train_loader, test_loader, epochs=EPOCHS)

# Run Active Learning with Random Sampling (for comparison)
active_learning_loop(model, train_loader, test_loader, unlabeled_dataset, strategy="random")

# Run Active Learning with Uncertainty Sampling
active_learning_loop(model, train_loader, test_loader, unlabeled_dataset, strategy="uncertainty")