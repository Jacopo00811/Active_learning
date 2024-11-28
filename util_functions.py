import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def setup_model(device, model_name, pretrained_weights):
    """
    Setup a fresh model instance
   
    Args:
        device (torch.device): Device to use for the model
        model_name (str): Name of the model to use
        pretrained_weights (bool): Whether to use pretrained weights
            
    Returns:
        model (nn.Module): Fresh model instance
    """
    if model_name == "ResNet-18":
        # Load pretrained weights if specified, otherwise None
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained_weights else None
        
        # Create a new model instance
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)
        return model
    else:
        raise ValueError(f"Model {model_name} not supported.")

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

def train_model(device, model, epochs, train_loader, val_loader, generator=None):
    """
    Train model on the training set and evaluate on the validation set for a specified number of epochs.

    Args:
        device (torch.device): Device to use for training and evaluation
        model (torch.nn.Module): Model to train and evaluate
        epochs (int): Number of epochs to train the model
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set
    """
    if generator is not None:
        torch.manual_seed(generator.initial_seed())

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=hyperparameters['weight_decay'])

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
        
        print(f"    Epoch {epoch+1}/{epochs}, Validation: {val_accuracy:.2f}% acc", end='\r') # Print epoch information, overwrite the line for each epoch
    print(end='\x1b[2K\r')  # Clear the line after training is finished

def full_training_set_baseline(device, model_name, pretrained_weights, epochs, batch_size, train_dataset, val_dataset, test_dataset, generator):
    # Create a new generator specifically for this run. This ensures we start from the same state every time
    run_generator = torch.Generator()
    run_generator.manual_seed(generator.initial_seed())

    # Create dataloaders with worker_init_fn to ensure worker randomness is controlled
    def seed_worker(worker_id):
        # Each worker needs its own seed
        worker_seed = run_generator.initial_seed() + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    full_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
    full_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
    full_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)

    # Initialize model
    model=setup_model(
        device=device, 
        model_name=model_name, 
        pretrained_weights=pretrained_weights
    )

    # Train model on full training set and evaluate on full validation set
    train_model(
        device=device, 
        model=model, 
        epochs=epochs, 
        train_loader=full_train_loader, 
        val_loader=full_val_loader, 
        generator=run_generator
    )

    # Evaluate model on full test set
    test_accuracy = evaluate_model(
        device=device, 
        model=model, 
        data_loader=full_test_loader
    )

    return test_accuracy


def active_learning_loop(
    device, 
    model_name,
    pretrained_weights,
    epochs,
    batch_size,
    train_val_ratio,  
    train_val_dataset, 
    test_dataset, 
    generator,
    budget_initial_size,
    budget_query_size,
    budget_al_iterations,
    al_algorithm
):
    """
    Active Learning Loop with AL-query --> Reset Model & Train iterations.
    First iteration is done without AL step to train on initial labelled training and validation subsets.
    Subsequent num_train_al_iterations iterations query the AL strategy to select samples to label and add to training and validation subsets, then train the model on the updated labelled training and validation subset.


    Args:
        device (torch.device): Device to use for training and evaluation
        model_name (str): Name of the model to use
        pretrained_weights (bool): Whether to use pretrained weights
        epochs (int): Number of epochs to train the model in each train/AL iteration
        batch_size (int): Batch size for data loaders
        train_val_ratio (float): Ratio of labelled training set size to labelled validation set size
        train_val_dataset (torch.utils.data.Dataset): Full train/val dataset
        test_dataset (torch.utils.data.Dataset): Full test dataset
        generator (torch.Generator): Generator for random number generation
        budget_initial_size (int): Initial number of labelled samples to start with
        budget_query_size (int): Number of samples to query in each AL iteration
        budget_al_iterations (int): Number of AL iterations to run
        al_algorithm (str): Active Learning strategy to use ("uncertainty" or "random"), more will be added in the future

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
            3. Re-initialize model to reset weights to initial state, train the model on the labelled training set for epochs=epochs and evaluate on the validation set.
            4. After training is finished, evaluate the model on the test set and store the test accuracy + size of the labelled training and validation sets.
    """

    # Create a new generator specifically for this AL run. This ensures we start from the same state every time
    run_generator = torch.Generator()
    run_generator.manual_seed(generator.initial_seed())

    # Create a numpy random state that will stay consistent
    np_rng = np.random.RandomState(generator.initial_seed())

    # Create dataloaders with worker_init_fn to ensure worker randomness is controlled
    def seed_worker(worker_id):
        # Each worker needs its own seed
        worker_seed = run_generator.initial_seed() + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Store initial random states
    initial_torch_state = torch.get_rng_state()
    initial_np_state = np.random.get_state()
    if torch.cuda.is_available():
        initial_cuda_state = torch.cuda.get_rng_state()
    
    # Assert that generators are properly initialized
    assert run_generator.initial_seed() == generator.initial_seed(), \
        "Run generator not initialized with correct seed"


    # Create dataloader for the full test dataset
    full_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)

    # Initialize labelled and unlabelled datasets using our controlled random state
    labelled_indices_global = np_rng.choice(
        len(train_val_dataset), 
        budget_initial_size, 
        replace=False
    )
    unlabelled_indices_global = list(set(range(len(train_val_dataset))) - set(labelled_indices_global))

    # Split the labelled dataset into train and validation
    train_size = int(len(labelled_indices_global) * train_val_ratio)
    val_size = len(labelled_indices_global) - train_size

    # Convert to tensor for torch splitting
    indices_tensor = torch.tensor(labelled_indices_global)
    splits = random_split(
        indices_tensor, 
        [train_size, val_size], 
        generator=run_generator
    )
    # Extract indices from the Subset objects
    labelled_train_indices_global = [indices_tensor[i].item() for i in splits[0].indices]
    labelled_val_indices_global = [indices_tensor[i].item() for i in splits[1].indices]

    # Ensure no overlap between labelled and unlabelled sets
    assert len(set(labelled_indices_global).intersection(set(unlabelled_indices_global))) == 0, "Data leakage detected between labelled and unlabelled sets"

    # Ensure no overlap between training and validation sets
    assert len(set(labelled_train_indices_global).intersection(set(labelled_val_indices_global))) == 0, "Data leakage detected between training and validation sets"

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
    # (Might be needed for some AL-algorithms) Create labelled subset (both train and val together) with relative indices from the full train/val dataset using global indices
    labelled_dataset_relative = Subset(
        train_val_dataset, 
        labelled_indices_global
    )

    # Initialize labelled train and validation dataset loaders from relative subsets
    labelled_train_loader_relative = DataLoader(labelled_train_dataset_relative, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=run_generator,  # Use run_generator instead of generator
        worker_init_fn=seed_worker,  # Add the worker seeding function
        num_workers=0  # Important: Set to 0 for complete reproducibility)
    )
    labelled_val_loader_relative = DataLoader(labelled_val_dataset_relative, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
    # Initialize unlabelled dataset loader from relative subset
    unlabelled_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
    # (Might be needed for some AL-algorithms) Initialize labelled dataset (both train and val together) loader from relative subset
    labelled_loader_relative = DataLoader(labelled_dataset_relative, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)

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
    for i in range(budget_al_iterations+1):
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
                    budget_query_size=budget_query_size
                )
            elif al_algorithm == "random":
                selected_unlabelled_relative_indices = random_sampling(
                    unlabelled_loader_relative=unlabelled_loader_relative, 
                    budget_query_size=budget_query_size,
                    random_state=np_rng
                )
            elif al_algorithm == "typiclust":
                selected_unlabelled_relative_indices = typiclust_sampling(
                    model=model, 
                    typiclust_obj=typiclust,
                    unlabelled_loader_relative=unlabelled_loader_relative,
                    budget=budget_query_size
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
            selected_unlabelled_global_indices = np_rng.permutation(selected_unlabelled_global_indices)

            # Split the selected global indices into selected labelled training and validation global indices based on the train/val ratio (bias was removed by shuffling selected global indices)
            selected_labelled_train_size = int(len(selected_unlabelled_global_indices) * train_val_ratio)
            selected_labelled_train_indices_global = selected_unlabelled_global_indices[:selected_labelled_train_size]
            selected_labelled_val_indices_global = selected_unlabelled_global_indices[selected_labelled_train_size:]

            # Update the labelled training and validation global indices with the selected global indices
            labelled_train_indices_global = np.concatenate([labelled_train_indices_global, selected_labelled_train_indices_global])
            labelled_val_indices_global = np.concatenate([labelled_val_indices_global, selected_labelled_val_indices_global])
            # (Might be needed for some AL-algorithms) Update the labelled global indices with the selected global indices
            labelled_indices_global = np.concatenate([labelled_train_indices_global, labelled_val_indices_global])

            # Ensure no overlap between labelled and unlabelled sets
            assert len(set(labelled_indices_global).intersection(set(unlabelled_indices_global))) == 0, "Data leakage detected between labelled and unlabelled sets"

            # Ensure no overlap between training and validation sets
            assert len(set(labelled_train_indices_global).intersection(set(labelled_val_indices_global))) == 0, "Data leakage detected between training and validation sets"

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
            # (Might be needed for some AL-algorithms) Create labelled subset (both train and val together) with relative indices from the full train/val dataset using global indices
            labelled_dataset_relative = Subset(
                train_val_dataset, 
                labelled_indices_global
            )

            # Re-initialize labelled train and validation dataset loaders from updated relative subsets
            labelled_train_loader_relative = DataLoader(labelled_train_dataset_relative, batch_size=batch_size, shuffle=True, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
            labelled_val_loader_relative = DataLoader(labelled_val_dataset_relative, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
            # Re-initialize unlabelled dataset loader from updated relative subset
            unlabelled_loader_relative = DataLoader(unlabelled_dataset_relative, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
            # (Might be needed for some AL-algorithms) Re-nitialize labelled dataset (both train and val together) loader from relative subset
            labelled_loader_relative = DataLoader(labelled_dataset_relative, batch_size=batch_size, shuffle=False, drop_last=False, generator=run_generator, worker_init_fn=seed_worker, num_workers=0)
            
            # Verify sample sizes are as expected
            expected_size = budget_initial_size + (i * budget_query_size)
            actual_size = len(labelled_train_indices_global) + len(labelled_val_indices_global)
            assert actual_size == expected_size, \
                f"Iteration {i}: Expected {expected_size} samples, got {actual_size}"
            
            # Verify no duplicates in selected samples
            assert len(set(selected_unlabelled_global_indices)) == len(selected_unlabelled_global_indices), \
                "Duplicate samples selected by AL algorithm"
            
        else: # Runs for the first iteration
            al_algorithm_time_iter = "None"

            # Initialize typiclust object if typiclust AL algorithm is selected
            if al_algorithm == "typiclust":
                typiclust = Typiclust(initial_labeled_size=budget_initial_size, device=device, n_components=50)



        ### Train Model on Labelled Data and Evaluate on Test Data ###

        # Re-initialize model with initial state
        model = setup_model(
            device=device, 
            model_name=model_name, 
            pretrained_weights=pretrained_weights
        )

        # Train model on labelled training subset and evaluate on labelled validation subset
        training_time_iter = time.time()
        train_model(
            device=device, 
            model=model, 
            epochs=epochs,
            train_loader=labelled_train_loader_relative, 
            val_loader=labelled_val_loader_relative,
            generator=run_generator
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
        
        print(f"  Iteration {i}/10 - Samples: {train_val_set_sizes[i]} ({(1 - train_val_ratio)*100:.0f}% val), Test: {test_accuracy:.2f}% acc, Time: {format_time(iter_time)}, Training: {format_time(training_time_iter)}, AL: {format_time(al_algorithm_time_iter) if al_algorithm_time_iter != 'None' else 'N/A'}")

        # Verify random states haven't unexpectedly changed
        assert torch.equal(torch.get_rng_state(), initial_torch_state), \
            "PyTorch random state changed unexpectedly"
        assert np.array_equal(np.random.get_state()[1], initial_np_state[1]), \
            "NumPy random state changed unexpectedly"
        if torch.cuda.is_available():
            assert torch.equal(torch.cuda.get_rng_state(), initial_cuda_state), \
                "CUDA random state changed unexpectedly"

    return train_val_set_sizes, test_accuracies, training_time_total, al_algorithm_time_total



