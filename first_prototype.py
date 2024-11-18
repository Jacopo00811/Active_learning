import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random


# To-do:
# * Track the time for each iteration, AL-algorithm run, budget strategy run and the entire run, store in the end.
# * Implement new AL-algorithms
# * Implement ability to run multiple runs with different seeds. MAKE SURE RANDOM INITIALIZATION WORKS PROPERLY, TEST EACH ALGORITHM WITH RANDOM INITIALIZATION FOR EACH RUN AND MAKE SURE IT IS CONSISTENT. Needs to dynamically switch over to average
# * For HPC: Do 5 runs, each with different seeds, for each budget strategy and AL-algorithm using, and then average them out, to ensure that the differences in the AL-algorithm results are not due to local optima in the randomnes. Plot the average alongg with the standard deviation.


### AL parameters ###
seeds = [0] # Set random seed to decrease uncertainty
al_algorithms = ['random', 'uncertainty'] # Active Learning algorithms to run
run_all_labelled_baseline = False # Enable to run the model with all labelled data to establish maximum performance baseline


## Budget Strategies ##
NUM_ITERATIONS = 20 # Number of iterations (budget increases) to run for each budget strategy
INITIAL_LABEL_SIZE_TO_BATCH_SIZE_RATIO = 3
enable_budget_strategies = [True, False, False, False] # Enable/disable budget strategies (1, 2, 3, 4)
strategy_label_batch_sizes = [200, 400, 800, 1600] # Labels added per iteration for each budget strategy, corresponding to 10%, 20%, 40%, 80% of the dataset labelled after 20 iterations depending on the budget strategy
strategy_initial_labelled_dataset_sizes = [INITIAL_LABEL_SIZE_TO_BATCH_SIZE_RATIO*label_batch_size for label_batch_size in strategy_label_batch_sizes] # Initial budget strategy labelled dataset sizes are set to 3x the label batch size


### Model Hyperparameters ###
TRAIN_VAL_RATIO = 0.8 # Ratio of training data to use for training (remaining is used for validation)
EPOCHS = 5 # Number of epochs to train the model for each budget size
epoch_training_status = False # Prints epochs + validation accuracy. Set to False for cleaner output



# Define relative folder and file path
relative_folder = "./run_results"

# Preset abbreviations for algorithms
algorithm_abbreviations = {
    'random': 'ran',
    'uncertainty': 'unc',
    'margin': 'mar',      # Example additional algorithm
    'entropy': 'ent'      # Example additional algorithm
}

all_rounds_data = {}
active_budget_strategies = [i for i, value in enumerate(enable_budget_strategies) if value]

all_rounds_data["config"] = {
    "seeds": seeds,
    "al_algorithms": al_algorithms,
    "num_iterations": NUM_ITERATIONS,
    "initial_label_size_to_batch_size_ratio": INITIAL_LABEL_SIZE_TO_BATCH_SIZE_RATIO,
    "budget_strategies": [x + 1 for x in active_budget_strategies],
    "enable_budget_strategies": enable_budget_strategies,
    "strategy_label_batch_sizes": strategy_label_batch_sizes,
    "strategy_initial_labelled_dataset_sizes": strategy_initial_labelled_dataset_sizes,
    "TRAIN_VAL_RATIO": TRAIN_VAL_RATIO,
    "EPOCHS": EPOCHS
}


# Set random seed for reproducibility
torch.manual_seed(seeds[0])
np.random.seed(seeds[0])
random.seed(seeds[0])
generator = torch.Generator()
generator.manual_seed(seeds[0])

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
train_size = int(TRAIN_VAL_RATIO * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

# Data loaders
full_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator) # Only used for training with all labels
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=generator)


# Reset model weights
def reset_model_weights(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Training function
def train_model(model, train_loader, val_loader, epochs=EPOCHS, epoch_training_status=True):
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
        if epoch_training_status:
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
    train_model(model, train_loader, val_loader, epochs=epochs, epoch_training_status=epoch_training_status)
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
    labelled_train_loader_relative = DataLoader(labelled_dataset_relative, batch_size=64, shuffle=True, generator=generator)
    unlabelled_train_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=64, shuffle=False, generator=generator)
    accuracies = []
    training_set_sizes = []

    # Run AL training loop for num_iterations iterations
    for i in range(num_iterations+1):
        # Reset model weights at the start of each iteration
        reset_model_weights(model)

        train_model(model, labelled_train_loader_relative, val_loader, epochs=EPOCHS, epoch_training_status=epoch_training_status)
        test_accuracy = evaluate_model(model, test_loader)
        accuracies.append(test_accuracy)
        training_set_sizes.append(len(labelled_dataset_relative))

        print(f"    AL Iteration {i}/{num_iterations}, Current Train Set Size: {training_set_sizes[i]}, Test Accuracy: {test_accuracy:.2f}%")

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

        labelled_train_loader_relative = DataLoader(labelled_dataset_relative, batch_size=64, shuffle=True, generator=generator)
        unlabelled_train_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=64, shuffle=False, generator=generator)

    return training_set_sizes, accuracies


# (If enabled) Train the model with entire dataset labelled to establish maximum performance baseline
if run_all_labelled_baseline:
    baseline_full_data_performance(model, full_train_loader, val_loader, test_loader, epochs=EPOCHS)

# Run full AL run (20 iterations for each l0, batch_size config) for each AL-strategy
# For each budget strategy, visualize and compare all AL strategies
for i in active_budget_strategies:
    initial_labelled_dataset_size = strategy_initial_labelled_dataset_sizes[i]
    label_batch_size = strategy_label_batch_sizes[i]

    budget_size =  initial_labelled_dataset_size + label_batch_size*NUM_ITERATIONS

    print(f"\nBudget Strategy {i+1} - Total Budget size: {budget_size}, Iterations: {NUM_ITERATIONS}, Initial Size: {initial_labelled_dataset_size}, Label Batch Size: {label_batch_size}")

    all_rounds_data[f"budget_strategy_{i+1}"] = {}

    for al_strategy in al_algorithms:
        print(f"  Running Active Learning Algorithm: {al_strategy.capitalize()}...")

        all_rounds_data[f"budget_strategy_{i+1}"][al_strategy] = {}
        
        # Run active learning loop, return training set sizes and corresponding test accuracies
        training_set_sizes, accuracies = active_learning_loop(model=model, num_iterations=NUM_ITERATIONS, initial_labelled_dataset_size=initial_labelled_dataset_size, label_batch_size=label_batch_size, al_algorithm=al_strategy)

        # Store
        all_rounds_data[f"budget_strategy_{i+1}"][al_strategy]["training_set_sizes"] = training_set_sizes
        all_rounds_data[f"budget_strategy_{i+1}"][al_strategy]["accuracies"] = accuracies
        
        print(f"    AL-Algorithm: {al_strategy.capitalize()}, Final Test Performance: {accuracies[-1]:.2f}%")

    print(f"\nBudget Strategy {i+1} Completed, Comparing AL-Algorithms & Visualizing Results...")
    plt.figure()
    for al_strategy in al_algorithms:
        training_set_size = all_rounds_data[f"budget_strategy_{i+1}"][al_strategy]["training_set_sizes"]
        accuracies = all_rounds_data[f"budget_strategy_{i+1}"][al_strategy]["accuracies"]
        plt.plot(training_set_sizes, accuracies, label=f"{al_strategy.capitalize()}",marker='o')
        print(f"    AL-Algorithm: {al_strategy.capitalize()}, Final Test Performance: {accuracies[-1]:.2f}%")
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Active Learning: Labelled Training Set Size VS Test Accuracy - Budget Strategy {i+1})')
    plt.grid(True)
    plt.legend()
    plt.show()


# Map the present algorithms to their abbreviations
selected_abbreviations = [algorithm_abbreviations[algo] for algo in al_algorithms if algo in algorithm_abbreviations]

# Construct the filename
file_name = (
    f"seeds_{'_'.join(map(str, seeds))}_"
    f"iter_{NUM_ITERATIONS}_"
    f"initbatchratio_{INITIAL_LABEL_SIZE_TO_BATCH_SIZE_RATIO}_"
    f"budget_{'_'.join(map(str, [x + 1 for x in active_budget_strategies]))}_"
    f"{'_'.join(selected_abbreviations)}_trainsplit_"
    f"{str(TRAIN_VAL_RATIO).replace('.', 'point')}_"
    f"epochs_{EPOCHS}"
)

file_path = os.path.join(relative_folder, file_name)

# Ensure the folder exists
os.makedirs(relative_folder, exist_ok=True)

# Save the dictionary as a JSON file
with open(file_path, "w") as file:
    json.dump(all_rounds_data, file, indent=4)

print(f"Data {file_name} saved to {file_path}")


