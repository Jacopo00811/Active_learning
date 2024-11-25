import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
import time
from al_algorithms import *
from Typiclust import Typiclust

def format_time(seconds):
    """
    Format time in seconds to a human-readable string in format "Xh Ym Zs"
    
    Args:
        seconds (float or None): Time in seconds to format
        
    Returns:
        str: Formatted time string showing only non-zero units
            - Returns 'N/A' if seconds is None
            
    Examples:
    >>> format_time(45.67)
    '45.67s'
    >>> format_time(125.5)
    '2m 5.50s'
    >>> format_time(7384.2)
    '2h 3m 4.20s'
    >>> format_time(None)
    'N/A'
    """

    if seconds is None:
        return 'N/A'
        
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"


def set_global_seed(seed):
    """
    Set global seed for reproducibility in PyTorch, NumPy, and Python random.
    Ensures that random number generation is consistent across runs, reducing uncertainty in results.

    Args:
        seed (int): Seed to set for random number generation
    
    Returns:
        torch.Generator: Generator for use in data loaders
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def reset_model_weights(model):
    """
    Reset model weights to default initialization.
    Is done for each train/AL iteration to ensure consistent starting point for model training each time the labelled training and validation subsets is updated.
    
    Args:
        model (torch.nn.Module): Model to reset weights for
    """

    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def evaluate_model(device, model, data_loader):
    """
    Evaluate model on a dataset and return the accuracy.

    Args:
        device (torch.device): Device to use for evaluation
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on
    """

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

def train_model(device, model, epochs, train_loader, val_loader, print_iteration_epochs=False):
    """
    Train model on the training set and evaluate on the validation set for a specified number of epochs.

    Args:
        device (torch.device): Device to use for training and evaluation
        model (torch.nn.Module): Model to train and evaluate
        epochs (int): Number of epochs to train the model
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set
        print_iteration_epochs (bool): Whether to print epoch information during training
    """

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

        val_accuracy = evaluate_model(
            device=device, 
            model=model, 
            data_loader=val_loader
        )
        
        if print_iteration_epochs:
            print(f"    Epoch {epoch+1}/{epochs}, Validation: {val_accuracy:.2f}% acc")

def active_learning_loop(device, model, epochs, train_val_dataset, train_val_ratio, full_test_loader, generator, num_train_al_iterations, initial_label_size, label_batch_size, al_algorithm, print_al_iterations=False, print_iteration_epochs=False):
    """
    Active Learning Loop with AL-query --> Reset Model & Train iterations.
    First iteration is done without AL step to train on initial labelled training and validation subsets.
    Subsequent num_train_al_iterations iterations query the AL strategy to select samples to label and add to training and validation subsets, then train the model on the updated labelled training and validation subset.


    Args:
        device (torch.device): Device to use for training and evaluation
        model (torch.nn.Module): Model to train and evaluate
        epochs (int): Number of epochs to train the model in each train/AL iteration
        train_val_dataset (torch.utils.data.Dataset): Full train/val dataset
        train_val_ratio (float): Ratio of labelled training set size to labelled validation set size
        full_test_loader (torch.utils.data.DataLoader): DataLoader for the full test dataset
        generator (torch.Generator): Generator for random number generation
        num_train_al_iterations (int): Number of train/AL iterations to run
        initial_label_size (int): Initial number of labelled samples split into training and validation subsets
        label_batch_size (int): Number of samples to label in each AL iteration to split between and add to training and validation subsets
        al_algorithm (str): Active Learning strategy to use ("uncertainty" or "random"), more will be added in the future
        print_al_iterations (bool): Whether to print iteration information
        print_iteration_epochs (bool): Whether to print epoch information during training

    Returns:
        tuple: Tuple containing:
            train_val_set_sizes (list): List of training + validation set sizes after each iteration
            test_accuracies (list): List of test accuracies after each iteration
            training_time_total (float): Total training time across all iterations
            al_algorithm_time_total (float): Total AL algorithm time across all iterations

            
    In-depth Explanation of data subset handling, initialization and active learning loop:        
        Explanation for data subset handling:
            The full train/val dataset has the global indices for the entire dataset.
            When we create labelled (train and val) and unlabelled subsets of the full train/val dataset, the indices are relative to that subset.
            As we are always working with two subsets (labelled and unlabelled), which need to be updated after each Train/AL iteration, we need to keep track of what global indices in the full train/val dataset the relative indices in the subsets correspond to.
        
        The initial labelled (training and validation) and unlaballed data subsets setup is as follows:
            1. Get global indices for labelled and unlabelled datasets from the full train/val dataset.
                1.1 Get labelled global indices by randomly selecting initial_label_size global indices.
                1.2 Get unlabelled global indices by removing the labelled global indices from the full train/val dataset indices.
            2. Randomly split the labelled global indices into training and validation global indices based on the train/val ratio.
            3. Create initial training, validation and unlabelled subsets with relative indices from the full train/val dataset using the global indices.
            4. Initialize labelled and unlabelled dataset loaders from the initial relative subsets.
        

        The training/active learning loop runs for num_train_al_iterations iterations, where each iteration consists of the following steps:
            1. On first iteration, skip the AL step and train the model on the initial labelled training and validation subsets.
            2. On subsequent iterations, query the AL strategy to select relative indices from the unlabelled subset to move to the labelled training and validation subset.
                2.1. Select relative indices from unlabelled subset to label based on the selected AL strategy.
                2.2 Convert the AL-selected relative indices to global indices from the full train/val dataset.
                2.3 Remove the selected global indices from the unlabelled global indices.
                2.4 Update the map from relative unlabelled indices to global indices (since the size of the unlabelled dataset has changed, we need to re-index the relative indices and remove the selected global indices).
                2.5 Shuffle the selected global indices to avoid bias (some AL-algorithms might return prioritized indices).
                2.6 Split the selected global indices into selected labelled training and validation global indices based on the train/val ratio.
                2.7 Update the labelled training and validation global indices with the selected global indices.
                2.8 Create labelled training and validation subsets with relative indices from the full train/val dataset using the updated (added) global indices.
                2.9 Create unlabelled subset with relative indices from the full train/val dataset using the updated (removed) global indices.
                2.10 Re-initialize labelled train and validation dataset loaders from the updated relative subsets.
                2.11 Re-initialize unlabelled dataset loader from the updated relative subset.
                2.12 Train the model on the labelled training set for epochs=epochs and evaluate on the validation set.
                2.13 After training is finished, evaluate the model on the test set and store the test accuracy.
            3. Reset model weights, train the model on the labelled training set for epochs=epochs and evaluate on the validation set.
            4. After training is finished, evaluate the model on the test set and store the test accuracy + size of the labelled training and validation sets.
            5. Print iteration information if print_al_iterations is True.
    """

    # Initialize labelled and unlabelled datasets
    labelled_indices_global = np.random.choice(len(train_val_dataset), initial_label_size, replace=False)
    unlabelled_indices_global = list(set(range(len(train_val_dataset))) - set(labelled_indices_global))

    # Split the labelled dataset into train and validation
    train_size = int(len(labelled_indices_global) * train_val_ratio)
    val_size = len(labelled_indices_global) - train_size
    labelled_train_indices_global, labelled_val_indices_global = random_split(labelled_indices_global, [train_size, val_size], generator=generator)

    # Create labelled training and validation subsets with relative indices from the full train/val dataset using global indices
    labelled_train_dataset_relative = Subset(
        train_val_dataset, 
        labelled_train_indices_global
    )
    labelled_val_dataset_relative = Subset(
        train_val_dataset, 
        labelled_val_indices_global
    )
    # Create unlabelled subset with relative indices from the full train/val dataset using global indices
    unlabelled_dataset_relative = Subset(
        train_val_dataset, 
        unlabelled_indices_global
    )

    # Initialize labelled train and validation dataset loaders from relative subsets
    labelled_train_loader_relative = DataLoader(labelled_train_dataset_relative, batch_size=64, shuffle=True, drop_last=False, generator=generator)
    labelled_val_loader_relative = DataLoader(labelled_val_dataset_relative, batch_size=64, shuffle=False, drop_last=False, generator=generator)
    # Initialize unlabelled dataset loader from relative subset
    unlabelled_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=64, shuffle=False, drop_last=False, generator=generator)

    # Define a map from unlabelled relative indices to unlabelled global indices
    # Used to move unlabelled AL-selected relative indices to equivalent global indices
    unlabelled_relative_to_global_indices_map = {relative_idx: global_idx for relative_idx, global_idx in enumerate(unlabelled_indices_global)}


    # Initialize lists to store test accuracies and training + validation set sizes
    test_accuracies = []
    train_val_set_sizes = []
    
    # Initialize total training time and AL algorithm time
    training_time_total = 0
    al_algorithm_time_total = 0

    # Initialize Typiclust object if typiclust AL algorithm is selected
    typiclust = None

    # Loop through active learning iterations
    for i in range(num_train_al_iterations+1):
        iter_time = time.time()
        

        ### Active Learning Step ###
        # Skip AL step for the first iteration, as we already have the initial labelled and unlabelled datasets
        if i > 0:
            al_algorithm_time_iter = time.time()

            # Select indices to label based on selected AL strategy
            if al_algorithm == "uncertainty":
                selected_unlabelled_relative_indices = uncertainty_sampling_least_confidence(
                    device=device, 
                    model=model, 
                    unlabelled_loader_relative=unlabelled_loader_relative, 
                    label_batch_size=label_batch_size
                )
            elif al_algorithm == "random":
                selected_unlabelled_relative_indices = random_sampling(
                    unlabelled_loader_relative=unlabelled_loader_relative, 
                    label_batch_size=label_batch_size
                )
            elif al_algorithm == "typiclust":
                selected_unlabelled_relative_indices = typiclust_sampling(
                    model=model, 
                    typiclust_obj=typiclust,
                    unlabelled_loader_relative=unlabelled_loader_relative,
                    budget=label_batch_size
                )
            else:
                raise ValueError("Unsupported strategy!")

            al_algorithm_time_iter = time.time() - al_algorithm_time_iter
            al_algorithm_time_total += al_algorithm_time_iter


            ### Update labelled and unlabelled datasets for the next iteration ###
            # Convert AL-selected unlabelled relative indices to unlabelled global indices from full train/val dataset
            selected_unlabelled_global_indices = [unlabelled_relative_to_global_indices_map[relative_idx] for relative_idx in selected_unlabelled_relative_indices]

            # Remove the selected global indices from the unlabelled global indices
            unlabelled_indices_global = list(set(unlabelled_indices_global) - set(selected_unlabelled_global_indices))

            # Update the map from relative unlabelled indices to global indices (since the size of the unlabelled dataset has changed)
            unlabelled_relative_to_global_indices_map = {relative_idx: global_idx for relative_idx, global_idx in enumerate(unlabelled_indices_global)}

            # Shuffle the selected global indices to avoid bias
            selected_unlabelled_global_indices = np.random.permutation(selected_unlabelled_global_indices)

            # Split the selected global indices into selected labelled training and validation global indices based on the train/val ratio (bias was removed by shuffling selected global indices)
            selected_labelled_train_size = int(len(selected_unlabelled_global_indices) * train_val_ratio)
            selected_labelled_train_indices_global = selected_unlabelled_global_indices[:selected_labelled_train_size]
            selected_labelled_val_indices_global = selected_unlabelled_global_indices[selected_labelled_train_size:]

            # Update the labelled training and validation global indices with the selected global indices
            labelled_train_indices_global = np.concatenate([labelled_train_indices_global, selected_labelled_train_indices_global])
            labelled_val_indices_global = np.concatenate([labelled_val_indices_global, selected_labelled_val_indices_global])

            # Create labelled training and validation subsets with relative indices from the full train/val dataset using updated (added) global indices
            labelled_train_dataset_relative = Subset(
                train_val_dataset, 
                labelled_train_indices_global
            )
            labelled_val_dataset_relative = Subset(
                train_val_dataset, 
                labelled_val_indices_global
            )
            # Create unlabelled subset with relative indices from the full train/val dataset using updated (removed) global indices
            unlabelled_dataset_relative = Subset(
                train_val_dataset, 
                unlabelled_indices_global
            )

            # Re-initialize labelled train and validation dataset loaders from updated relative subsets
            labelled_train_loader_relative = DataLoader(labelled_train_dataset_relative, batch_size=64, shuffle=True, drop_last=False, generator=generator)
            labelled_val_loader_relative = DataLoader(labelled_val_dataset_relative, batch_size=64, shuffle=False, drop_last=False, generator=generator)
            # Re-initialize unlabelled dataset loader from updated relative subset
            unlabelled_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=64, shuffle=False, drop_last=False, generator=generator)
        else:
            print("Initial Training on Labelled Data")
            al_algorithm_time_iter = "None"

            # Initialize typiclust object if typiclust AL algorithm is selected
            if al_algorithm == "typiclust":
                typiclust = Typiclust(initial_labeled_size=initial_label_size, device=device, n_components=50)


        ### Train Model on Labelled Data and Evaluate on Test Data ###

        # Reset model weights at the start of each iteration
        reset_model_weights(model)

        # Train model on labelled training subset and evaluate on labelled validation subset
        training_time_iter = time.time()
        train_model(
            device=device, 
            model=model, 
            epochs=epochs,
            train_loader=labelled_train_loader_relative, 
            val_loader=labelled_val_loader_relative, 
            print_iteration_epochs=print_iteration_epochs
        )
        
        training_time_iter = time.time() - training_time_iter
        training_time_total += training_time_iter

        # Evaluate model on full test dataset
        test_accuracy = evaluate_model(
            device=device, 
            model=model, 
            data_loader=full_test_loader
        )

        # Store test accuracy and training + validation set sizes
        test_accuracies.append(test_accuracy)
        train_val_set_sizes.append(len(labelled_train_dataset_relative) + len(labelled_val_dataset_relative))


        ### Print Iteration Information ###
        iter_time = time.time() - iter_time
        if print_al_iterations:
            print(f"  Iteration {i}/10 - Samples: {train_val_set_sizes[i]} ({(1 - train_val_ratio)*100:.0f}% val), Test: {test_accuracy:.2f}% acc, Time: {format_time(iter_time)}, Training: {format_time(training_time_iter)}, AL: {format_time(al_algorithm_time_iter) if al_algorithm_time_iter != 'None' else 'N/A'}")

    return train_val_set_sizes, test_accuracies, training_time_total, al_algorithm_time_total


def plot_al_performance_across_seeds(simulation_data, budget_strategy, al_algorithms, confidence_level=0.95):
    """
    Plot average performance of AL algorithms across seeds with confidence intervals.

    Args:
        simulation_data (dict): Simulation data containing results and configuration
        budget_strategy (str): Budget strategy used in simulation
        al_algorithms (list): List of AL algorithms to plot
        confidence_level (float): Confidence level for confidence intervals
    
    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the plot
    """

    # Plot average performance of AL algorithms across seeds with confidence intervals.
    plt.figure(figsize=(10, 6))
    
    train_val_ratio = simulation_data["config"]["model"]["train_val_ratio"]
    
    for idx, al_algorithm in enumerate(al_algorithms):
        # Collect data across all seeds
        all_train_val_set_sizes = []
        all_test_accuracies = []
        
        for seed_name in simulation_data["results"].keys():
            train_val_set_sizes = simulation_data["results"][seed_name][f"budget_strategy_{budget_strategy}"][al_algorithm]["train_val_set_sizes"]
            test_accuracies = simulation_data["results"][seed_name][f"budget_strategy_{budget_strategy}"][al_algorithm]["test_accuracies"]
            
            all_train_val_set_sizes.append(train_val_set_sizes)
            all_test_accuracies.append(test_accuracies)
        
        # Convert to numpy arrays for easier manipulation
        all_train_val_set_sizes = np.array(all_train_val_set_sizes)
        all_test_accuracies = np.array(all_test_accuracies)
        
        # Calculate mean and confidence interval
        mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
        std_test_accuracies = np.std(all_test_accuracies, axis=0)
        
        # Calculate confidence interval
        n_seeds = len(simulation_data["results"])
        confidence_interval = stats.t.ppf((1 + confidence_level) / 2, n_seeds - 1) * (std_test_accuracies / np.sqrt(n_seeds))
        
        # Plot mean line and confidence interval
        plt.plot(all_train_val_set_sizes[0], mean_test_accuracies, label=f"{al_algorithm.capitalize()}", marker='o', linewidth=2)
        plt.fill_between(all_train_val_set_sizes[0], 
                        mean_test_accuracies - confidence_interval,
                        mean_test_accuracies + confidence_interval,
                        alpha=0.2)
        
        # Print final performance statistics
        final_mean = mean_test_accuracies[-1]
        final_ci = confidence_interval[-1]
        print(f"    {al_algorithm.capitalize()}:")
        print(f"        Mean Final Accuracy: {final_mean:.2f}%")
        print(f"        95% CI: [{final_mean - final_ci:.2f}%, {final_mean + final_ci:.2f}%]")
    
    plt.xlabel(f'Training Set Size ({(1 - train_val_ratio) * 100:.0f}% for Validation)')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Active Learning Performance - Budget Strategy {budget_strategy}\n'
              f'(Average across {n_seeds} seeds with {int(confidence_level*100)}% confidence intervals)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add minor gridlines for better readability
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.minorticks_on()
    
    return plt.gcf()