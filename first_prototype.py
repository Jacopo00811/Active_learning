#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt



### AL parameters ###
seed = 0 # Consider defining a list of seeds
num_iterations = 20
budgets = 1 # Set to 4 for all 4 dataset initializations
label_batch_sizes = [200, 400, 800, 1600] # 10%, 20%, 40%, 80% of the dataset is labelled after 20 iterations
l0_sizes = [3*label_batch_size for label_batch_size in label_batch_sizes] # 3 times ...
al_strategies = ['random', 'uncertainty']
run_all_labelled_baseline = False


### Hyperparameters ###
TRAIN_SUBSET_RATIO = 0.8
EPOCHS = 5
debug = False # Shows epochs + validation accuracy


# Set random seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
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

# Data loaders
full_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Only used for training with all labels
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Reset model weights
def reset_model_weights(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Training function
def train_model(model, train_loader, val_loader, epochs=EPOCHS, debug=True):
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
        if debug:
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

# Function to train model on full data to establish baseline maximum performance
def baseline_full_data_performance(model, train_loader, val_loader, test_loader, epochs=EPOCHS):
    reset_model_weights(model)

    print("Training on the full dataset to find maximum performance baseline...")
    train_model(model, train_loader, val_loader, epochs=epochs, debug=debug)
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Maximum Performance Baseline - Test Accuracy: {test_accuracy:.2f}%")

# Uncertainty Sampling (Least Confidence)
def uncertainty_sampling_least_confidence(model, unlabelled_loader, label_batch_size):
    model.eval()
    uncertainties = []
    samples_indices = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(unlabelled_loader):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            max_confidences, _ = torch.max(softmax_outputs, dim=1)
            uncertainties.extend(1 - max_confidences.cpu().numpy())
            samples_indices.extend([idx] * images.size(0))

    uncertain_indices = np.argsort(uncertainties)[-label_batch_size:]
    selected_indices = [samples_indices[i] for i in uncertain_indices]
    return selected_indices

# Random Sampling Strategy
def random_sampling(unlabelled_loader, label_batch_size):
    indices = list(range(len(unlabelled_loader.dataset)))
    return np.random.choice(indices, label_batch_size, replace=False).tolist()

# Active Learning Loop with Query Strategy Selection and Model Reset
def active_learning_loop(model, num_iterations, l0_size, label_batch_size, al_strategy):
    # Initialize labelled and unlabelled datasets
    labelled_indices = np.random.choice(len(train_dataset), l0_size, replace=False)
    labelled_dataset = Subset(train_dataset, labelled_indices)

    unlabelled_indices = list(set(range(len(train_dataset))) - set(labelled_indices))
    unlabelled_dataset = Subset(train_dataset, unlabelled_indices)
    
    # Initialize labelled and unlabelled train loader
    labelled_train_loader = DataLoader(labelled_dataset, batch_size=64, shuffle=True)
    unlabelled_train_loader = DataLoader(unlabelled_dataset, batch_size=64, shuffle=False)
    accuracies = []
    training_set_sizes = []

    # Run AL training loop for num_iterations iterations
    for i in range(num_iterations):
        # Reset model weights at the start of each iteration
        reset_model_weights(model)

        train_model(model, labelled_train_loader, val_loader, epochs=EPOCHS, debug=debug)
        test_accuracy = evaluate_model(model, test_loader)
        accuracies.append(test_accuracy)
        training_set_sizes.append(len(labelled_dataset))

        print(f"    Budget {i+1}, Train set size: {training_set_sizes[i]}, Test Accuracy: {test_accuracy:.2f}%")

        # Select indices to label based on selected AL strategy
        if al_strategy == "uncertainty":
            selected_indices = uncertainty_sampling_least_confidence(model, unlabelled_train_loader, label_batch_size)
        elif al_strategy == "random":
            selected_indices = random_sampling(unlabelled_train_loader, label_batch_size)
        else:
            raise ValueError("Unsupported strategy!")

        # Label selected unlabelled indices, and redefine labelled and unlabelled train loaders with new subsets
        labelled_indices = np.concatenate([labelled_indices, selected_indices])
        labelled_dataset = Subset(train_dataset, labelled_indices)
        
        unlabelled_indices = list(set(unlabelled_indices) - set(selected_indices))
        unlabelled_dataset = Subset(train_dataset, unlabelled_indices)
        
        labelled_train_loader = DataLoader(labelled_dataset, batch_size=64, shuffle=True)
        unlabelled_train_loader = DataLoader(unlabelled_dataset, batch_size=64, shuffle=False)
    return training_set_sizes, accuracies


# (If enabled) Train the model with entire dataset labelled to establish maximum performance baseline
if run_all_labelled_baseline:
    baseline_full_data_performance(model, full_train_loader, val_loader, test_loader, epochs=EPOCHS)

# Run full AL run (20 iterations for each l0, batch_size config) for each AL-strategy
all_rounds = {}
for al_strategy in al_strategies:
    all_rounds[al_strategy] = {}
    for i in range(budgets):
        all_rounds[al_strategy][f"budget_{i+1}"] = {}

        # Initialize batch_size for
        l0_size = l0_sizes[i]
        label_batch_size = label_batch_sizes[i]

        print(f"Active Learning Strategy: {al_strategy}, Round: {i+1}, L0 Size: {l0_size}, Batch Size: {label_batch_size}")

        # Run active learning loop, return training set sizes and corresponding test accuracies
        training_set_sizes, accuracies = active_learning_loop(model=model, num_iterations=num_iterations, l0_size=l0_size, label_batch_size=label_batch_size, al_strategy=al_strategy)

        # Store
        all_rounds[al_strategy][f"budget_{i+1}"]["training_set_sizes"] = training_set_sizes
        all_rounds[al_strategy][f"budget_{i+1}"]["accuracies"] = accuracies

# For each budget size, visualize and compare all AL strategies
for i in range(budgets):
    budget_size = label_batch_sizes[i]*num_iterations + l0_sizes[i]
    print(f"\nBudget {i+1}, Total Budget size: {budget_size}, Iterations: {num_iterations}, Initial Size: {l0_sizes[i]}, Label Batch Size: {label_batch_sizes[i]}")
    plt.figure()
    for al_strategy in al_strategies:
        training_set_size = all_rounds[al_strategy][f"budget_{i+1}"]["training_set_sizes"]
        accuracies = all_rounds[al_strategy][f"budget_{i+1}"]["accuracies"]
        plt.plot(training_set_sizes, accuracies, label=f"{al_strategy.capitalize()}",marker='o')
        print(f"    AL-strategy: {al_strategy.capitalize()}, Final Test Performance: {accuracies[-1]:.2f}%")
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Active Learning: Labelled Training Set Size VS Test Accuracy - Round {i+1})')
    plt.grid(True)
    plt.legend()
    plt.show()

