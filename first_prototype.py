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
seed = 0 # Set random seed to decrease uncertainty. Consider defining a list of seeds
al_algorithms = ['random', 'uncertainty'] # Active Learning algorithms to run
run_all_labelled_baseline = False # Enable to run the model with all labelled data to establish maximum performance baseline

## Budget Strategies ##
num_iterations = 20 # Number of iterations (budget increases) to run for each budget strategy
enable_budget_strategies = [True, False, True, False] # Enable/disable budget strategies (1, 2, 3, 4)
strategy_label_batch_sizes = [200, 400, 800, 1600] # Labels added per iteration for each budget strategy, corresponding to 10%, 20%, 40%, 80% of the dataset labelled after 20 iterations depending on the budget strategy
strategy_initial_labelled_dataset_sizes = [3*label_batch_size for label_batch_size in strategy_label_batch_sizes] # Initial budget strategy labelled dataset sizes are set to 3x the label batch size


### Model Hyperparameters ###
TRAIN_SUBSET_RATIO = 0.8 # Ratio of training data to use for training (remaining is used for validation)
EPOCHS = 5 # Number of epochs to train the model for each budget size
debug = False # Prints epochs + validation accuracy. Set to False for cleaner output


# Set random seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
print(f"Model: {model.__class__.__name__}, Device: {device}\n")

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
def active_learning_loop(model, num_iterations, initial_labelled_dataset_size, label_batch_size, al_algorithm):
    # Define the global indices for the original dataset. Used to map labelled and unlabelled indices to the original dataset

    # Initialize labelled and unlabelled datasets
    labelled_indices_global = np.random.choice(len(train_dataset), initial_labelled_dataset_size, replace=False)
    labelled_dataset_relative = Subset(train_dataset, labelled_indices_global)

    unlabelled_indices_global = list(set(range(len(train_dataset))) - set(labelled_indices_global))
    unlabelled_dataset_relative = Subset(train_dataset, unlabelled_indices_global)

    # AL strategies only retrieve the relative indices of the unlabelled dataset (subset of full training dataset).
    # These relative indices need to be converted to the equivalent global indices in the full training dataset before they can be added to the labelled dataset.
    # Therefore we define a map from relative unlabelled indices to global indices.
    unlabelled_relative_to_global_indices_map = {relative_idx: global_idx for relative_idx, global_idx in enumerate(unlabelled_indices_global)}

    # Initialize labelled and unlabelled train loader
    labelled_train_loader_relative = DataLoader(labelled_dataset_relative, batch_size=64, shuffle=True)
    unlabelled_train_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=64, shuffle=False)
    accuracies = []
    training_set_sizes = []

    # Run AL training loop for num_iterations iterations
    for i in range(num_iterations):
        # Reset model weights at the start of each iteration
        reset_model_weights(model)

        train_model(model, labelled_train_loader_relative, val_loader, epochs=EPOCHS, debug=debug)
        test_accuracy = evaluate_model(model, test_loader)
        accuracies.append(test_accuracy)
        training_set_sizes.append(len(labelled_dataset_relative))

        print(f"    AL Iteration {i+1}/{num_iterations}, Current Train Set Size: {training_set_sizes[i]}, Test Accuracy: {test_accuracy:.2f}%")

        # Select indices to label based on selected AL strategy
        if al_algorithm == "uncertainty":
            selected_unlabelled_relative_indices = uncertainty_sampling_least_confidence(model, unlabelled_train_loader_relative, label_batch_size)
        elif al_algorithm == "random":
            selected_unlabelled_relative_indices = random_sampling(unlabelled_train_loader_relative, label_batch_size)
        else:
            raise ValueError("Unsupported strategy!")

        # Convert selected relative indices from unlabelled dataset to global indices from train_dataset
        selected_unlabelled_global_indices = [unlabelled_relative_to_global_indices_map[relative_idx] for relative_idx in selected_unlabelled_relative_indices]

        # Add the selected data points to the labelled dataset (which already has global indices)
        labelled_indices_global = np.concatenate([labelled_indices_global, selected_unlabelled_global_indices])
        labelled_dataset_relative = Subset(train_dataset, labelled_indices_global)

        # Remove the selected global indices from the unlabelled global indices
        unlabelled_indices_global = list(set(unlabelled_indices_global) - set(selected_unlabelled_global_indices))
        # Update the map from relative unlabelled indices to global indices (since the size of the unlabelled dataset has changed)
        unlabelled_relative_to_global_indices_map = {relative_idx: global_idx for relative_idx, global_idx in enumerate(unlabelled_indices_global)}
        # Update the unlabelled dataset with the new unlabelled indices
        unlabelled_dataset_relative = Subset(train_dataset, unlabelled_indices_global)

        labelled_train_loader_relative = DataLoader(labelled_dataset_relative, batch_size=64, shuffle=True)
        unlabelled_train_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=64, shuffle=False)

    return training_set_sizes, accuracies


# (If enabled) Train the model with entire dataset labelled to establish maximum performance baseline
if run_all_labelled_baseline:
    baseline_full_data_performance(model, full_train_loader, val_loader, test_loader, epochs=EPOCHS)

# Run full AL run (20 iterations for each l0, batch_size config) for each AL-strategy
all_rounds = {}
active_budget_strategies = [i for i, value in enumerate(enable_budget_strategies) if value]

# For each budget strategy, visualize and compare all AL strategies
for i in active_budget_strategies:
    initial_labelled_dataset_size = strategy_initial_labelled_dataset_sizes[i]
    label_batch_size = strategy_label_batch_sizes[i]

    budget_size =  initial_labelled_dataset_size + label_batch_size*num_iterations

    print(f"\nBudget Strategy {i+1}: Total Budget size: {budget_size}, Iterations: {num_iterations}, Initial Size: {initial_labelled_dataset_size}, Label Batch Size: {label_batch_size}")

    all_rounds[f"budget_strategy_{i+1}"] = {}

    for al_strategy in al_algorithms:
        print(f"  Running Active Learning Algorithm: {al_strategy.capitalize()}...")

        all_rounds[f"budget_strategy_{i+1}"][al_strategy] = {}
        
        # Run active learning loop, return training set sizes and corresponding test accuracies
        training_set_sizes, accuracies = active_learning_loop(model=model, num_iterations=num_iterations, initial_labelled_dataset_size=initial_labelled_dataset_size, label_batch_size=label_batch_size, al_algorithm=al_strategy)

        # Store
        all_rounds[f"budget_strategy_{i+1}"][al_strategy]["training_set_sizes"] = training_set_sizes
        all_rounds[f"budget_strategy_{i+1}"][al_strategy]["accuracies"] = accuracies
        
        print(f"    AL-Algorithm: {al_strategy.capitalize()}, Final Test Performance: {accuracies[-1]:.2f}%")

    print(f"\nBudget Strategy {i+1} Completed, Comparing AL-Algorithms & Visualizing Results...")
    plt.figure()
    for al_strategy in al_algorithms:
        training_set_size = all_rounds[f"budget_strategy_{i+1}"][al_strategy]["training_set_sizes"]
        accuracies = all_rounds[f"budget_strategy_{i+1}"][al_strategy]["accuracies"]
        plt.plot(training_set_sizes, accuracies, label=f"{al_strategy.capitalize()}",marker='o')
        print(f"    AL-Algorithm: {al_strategy.capitalize()}, Final Test Performance: {accuracies[-1]:.2f}%")
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Active Learning: Labelled Training Set Size VS Test Accuracy - Budget Strategy {i+1})')
    plt.grid(True)
    plt.legend()
    plt.show()

