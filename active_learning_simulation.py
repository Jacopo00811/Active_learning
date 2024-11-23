import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from util_functions import *




### Begin of Parameters ###

## Simulation Parameters ##
seeds = [0, 1] # Set random seeds to decrease uncertainty
train_full_dataset_baseline = True # Enable to run the model with all labelled data to establish maximum performance baseline
relative_val = True # Enable to use relative validation set size, otherwise use full validation set
save_results = True # Enable to save results to file
relative_save_folder = "./run_results" # Define relative save folder
print_al_iterations = True # Enable to print AL iteration results
print_iteration_epochs = False # Enable to print epochs + validation accuracy for each AL iteration

## Model Hyperparameters ##
TRAIN_VAL_RATIO = 0.8 # Ratio of training data to use for training (remaining is used for validation)
EPOCHS = 3 # Number of epochs to train the model for each budget size

## Active Learning Algorithm Parameters ##
NUM_TRAIN_AL_ITERATIONS = 10 # Number of Train/AL iterations to run for each algorithm for budget strategy
al_algorithms = ['random', 'uncertainty', 'typiclust'] # Active Learning algorithms to run
algorithm_abbreviations = { # Abbreviations for each algorithm, used for file naming
    'random': 'ran',
    'uncertainty': 'unc',
    'typiclust': 'typ',
    'margin': 'mar',      # Example additional algorithm, not implemented yet
    'entropy': 'ent'      # Example additional algorithm, not implemented yet
}

## Budget Strategy Parameters ##
RATIO_BATCH_TO_INIT_DATASET = 3 # Ratio of initial labelled dataset size to label batch size (labels added per iteration)
BUDGET_STRATEGIES = {
   1: {"active": True, "batch_size": 200},
   2: {"active": True, "batch_size": 400}, 
   3: {"active": False, "batch_size": 800},
   4: {"active": False, "batch_size": 1600}
}

### End of Parameters ###


# Setup budget strategies
selected_strategies = [num for num, config in BUDGET_STRATEGIES.items() if config["active"]]
label_batch_sizes = [BUDGET_STRATEGIES[num]["batch_size"] for num in selected_strategies]
initial_label_sizes = [BUDGET_STRATEGIES[num]["batch_size"] * RATIO_BATCH_TO_INIT_DATASET for num in selected_strategies]

# Calculate total number of baseline and training/AL iterations (used for status printing)
num_seeds = len(seeds)
num_strategies = len(selected_strategies)
num_al_algorithms = len(al_algorithms)
num_total_train_al_iterations = len(seeds) * len(selected_strategies) * len(al_algorithms) * (NUM_TRAIN_AL_ITERATIONS + 1) # + 1 because the first iteration uses the initial labelled dataset, so no active learning is used

# Set the device to CUDA / MPS if available, otherwise to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

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

# Store configuration data
simulation_data = {
   "config": {
       "simulation": {
           "device": str(device),
           "seeds": seeds,
           "train_full_dataset_baseline": train_full_dataset_baseline,
           "save_results": save_results,
           "relative_save_folder": relative_save_folder,
           "print_settings": {
               "al_iterations": print_al_iterations,
               "epochs": print_iteration_epochs
           }
       },
       "model": {
           "train_val_ratio": TRAIN_VAL_RATIO,
           "epochs": EPOCHS
       },
       "active_learning": {
           "train_al_iterations": NUM_TRAIN_AL_ITERATIONS,
           "al_algorithms": al_algorithms,
       },
       "budget": {
           "ratio_batch_to_init_dataset": RATIO_BATCH_TO_INIT_DATASET,
           "selected_strategies": {
                num: {
                    "batch_size": BUDGET_STRATEGIES[num]["batch_size"],
                    "initial_size": BUDGET_STRATEGIES[num]["batch_size"] * RATIO_BATCH_TO_INIT_DATASET,
                    "final_size": BUDGET_STRATEGIES[num]["batch_size"] * NUM_TRAIN_AL_ITERATIONS + BUDGET_STRATEGIES[num]["batch_size"] * RATIO_BATCH_TO_INIT_DATASET
                }
                for num in selected_strategies
           }
       }
   }
}

# Store runtime data
simulation_data["runtimes"] = {
    "simulation": None,
    "seeds": {f"seed_{seed}": None for seed in seeds},
    "full_dataset_baselines": {
        f"seed_{seed}_full_dataset_baseline": None
        for seed in seeds
    },
    "budget_strategies": {
        f"seed_{seed}_budget_{strategy}": None 
        for seed in seeds 
        for strategy in selected_strategies
    },
    "training_and_al_algorithms": {
        f"seed_{seed}_budget_{strategy}_{algo}": {"training": None, "al_algorithm": None}
        for seed in seeds
        for strategy in selected_strategies
        for algo in al_algorithms
    }
}

# Initialize results data
simulation_data["results"] = {}



print("==================== ACTIVE LEARNING SIMULATION ====================")
print("Model, Dataset and Device Configuration:")
print("- Dataset: CIFAR-10")
print("- Model: ResNet-18")
print("- Running on device:", device)
print(f"- Save Results: {save_results}")
print("<Dynamically print chosen Model and Dataset if/when multiple selectors have been implemented>")

print(f"\nAL Simulation Configuration:")
print(f"- Training/Validation Split Ratio: {TRAIN_VAL_RATIO}")
print(f"- Epochs per Training Iteration: {EPOCHS}")
print(f"- Seeds: {seeds}")
print(f"- Train Full Dataset Baseline: {train_full_dataset_baseline}, ")

print(f"\nActive Learning Algorithm Configuration:")
print(f"- Train/AL Iterations: {NUM_TRAIN_AL_ITERATIONS}")
print(f"- AL Algorithms: {', '.join(al_algorithms)}")

print(f"\nBudget Strategy Configuration:")
print(f"- Selected Budget Strategies: {selected_strategies}")
print(f"- Ratio of Initial Dataset Size to Label Batch Size: {RATIO_BATCH_TO_INIT_DATASET}x")
for i, strategy in enumerate(selected_strategies):
    print(f"  - Budget Strategy {strategy}:")
    print(f"    - Active: {BUDGET_STRATEGIES[strategy]['active']}")
    print(f"    - Label Batch Size: {label_batch_sizes[i]}")
    print(f"    - Initial Labelled Dataset Size: {initial_label_sizes[i]}")
    print(f"    - Final Labelled Dataset Size: {label_batch_sizes[i] * NUM_TRAIN_AL_ITERATIONS + initial_label_sizes[i]}")

print(f"\nNesting Structure of runs:")
print(f"- Number of Seeds to run: {num_seeds}")
print(f"  - Per seed: Baselines to run: {1 if train_full_dataset_baseline else 0}")
print(f"    - Per Baseline: Epochs to run: {EPOCHS}")
print(f"  - Per seed: Budget Strategies to run: {num_strategies}")
print(f"    - Per Budget Strategy: AL-Algorithms to run: {num_al_algorithms}")
print(f"      - Per AL-Algorithm: Train/AL Iterations to run: {NUM_TRAIN_AL_ITERATIONS}")
print(f"        - Per Train/AL Iteration: Epochs to run: {EPOCHS}")

print(f"\nTotal Baseline + Training/AL runs (of varying dataset sizes) & Epochs:")
print(f"- Number of baselines to train: {num_seeds}")
print(f"  - Number of Epochs: {num_seeds*EPOCHS}")
print(f"- Number of Train/AL-Iterations to run: {num_total_train_al_iterations}")
print(f"  - Number of Epochs: {num_total_train_al_iterations*EPOCHS}")
print(f"- Total Number of Epochs: {EPOCHS * (num_seeds + num_total_train_al_iterations)}")
print("====================================================================")

print("\nBeginning AL Simulation...")
print(f"Seeds to run: {seeds}")
print(f"Budget strategies per seed: {selected_strategies}")
print(f"AL-algorithms per budget strategy: {al_algorithms}")


# Initialize timer
simulation_time = time.time()

# Run the active learning evaluation loop for each seed
for seed_idx, seed in enumerate(seeds):
    seed_time = time.time()

    print(f"\n\n[SEED {seed} ({seed_idx+1}/{num_seeds})]")

    # Set random seed and generator used for data loaders
    generator = set_global_seed(seed)
    simulation_data["results"][f"seed_{seed}"] = {}

    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    # Data loaders
    full_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator) # Only used for training baseline with all labels
    full_val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, generator=generator) # Only used for validation baseline with all labels
    full_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=generator) # Both used for testing with all labels and for Train/AL iterations

    # Setup model
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    # print(f"Model: {model.__class__.__name__}")

    # (If enabled) Train the model with entire dataset labelled to establish maximum performance baseline
    if train_full_dataset_baseline:
        print("Training on the full dataset to find maximum performance baseline...")

        full_dataset_baseline_time = time.time()

        test_accuracy = train_model_full_dataset_baseline(
            device=device, 
            model=model, 
            epochs=EPOCHS,
            full_train_loader=full_train_loader, 
            full_val_loader=full_val_loader, 
            full_test_loader=full_test_loader, print_iteration_epochs=print_iteration_epochs
        )

        full_dataset_baseline_time = time.time() - full_dataset_baseline_time

        simulation_data["results"][f"seed_{seed}"]["full_dataset_baseline"] = test_accuracy
        simulation_data["runtimes"]["full_dataset_baselines"][f"seed_{seed}_full_dataset_baseline"] = full_dataset_baseline_time

        print(f"Baseline Complete - Test Accuracy: {test_accuracy:.2f}% - Runtime: {format_time(full_dataset_baseline_time)}")
    else:
        simulation_data["results"][f"seed_{seed}"]["full_dataset_baseline"] = None
        simulation_data["runtimes"]["full_dataset_baselines"][f"seed_{seed}_full_dataset_baseline"] = None
        print("Full dataset baseline disabled, skipping...")


    # For each budget strategy, train, evaluate, visualize and compare all AL strategies
    for strategy_idx, strategy in enumerate(selected_strategies):
        budget_strategy_time = time.time()

        label_batch_size = label_batch_sizes[strategy_idx]
        initial_label_size = initial_label_sizes[strategy_idx]
        final_label_size = label_batch_size * NUM_TRAIN_AL_ITERATIONS + initial_label_size

        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds})) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies})]")
        print(f"Initial size: {initial_label_size} → Final: {final_label_size} | Batch size: {label_batch_size}")

        simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"] = {}

        for al_algorithm_idx, al_algorithm in enumerate(al_algorithms):
            print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds}) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies}) | AL: {al_algorithm.upper()} ({al_algorithm_idx+1}/{num_al_algorithms})]")

            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm] = {}
            
            # Run active learning loop, return training set sizes and corresponding test accuracies
            train_val_set_sizes, test_accuracies, training_time_total, al_algorithm_time_total = active_learning_loop(
                device=device, 
                model=model, 
                epochs=EPOCHS, 
                train_val_dataset=train_val_dataset, 
                train_val_ratio=TRAIN_VAL_RATIO, 
                full_test_loader=full_test_loader, 
                generator=generator, 
                num_train_al_iterations=NUM_TRAIN_AL_ITERATIONS, 
                initial_label_size=initial_label_size, 
                label_batch_size=label_batch_size, 
                al_algorithm=al_algorithm, 
                print_al_iterations=print_al_iterations, 
                print_iteration_epochs=print_iteration_epochs
            )

            # Store results
            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["train_val_set_sizes"] = train_val_set_sizes
            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["test_accuracies"] = test_accuracies

            # Store training and AL algorithm runtimes
            simulation_data["runtimes"]["training_and_al_algorithms"][f"seed_{seed}_budget_{strategy}_{al_algorithm}"]["training"] = training_time_total
            simulation_data["runtimes"]["training_and_al_algorithms"][f"seed_{seed}_budget_{strategy}_{al_algorithm}"]["al_algorithm"] = al_algorithm_time_total

            print(f"{al_algorithm.upper()} complete - Test: {test_accuracies[-1]:.2f}% acc, Total training time: {format_time(training_time_total)}, Total AL time: {format_time(al_algorithm_time_total)}")


        budget_strategy_time = time.time() - budget_strategy_time
        simulation_data["runtimes"]["budget_strategies"][f"seed_{seed}_budget_{strategy}"] = budget_strategy_time

        strategy_best_algorithm = None
        strategy_best_accuracy = 0

        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds})) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies}) | COMPLETED]")
        for al_algorithm_idx, al_algorithm in enumerate(al_algorithms):
            test_accuracies = simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["test_accuracies"]

            if test_accuracies[-1] > strategy_best_accuracy or strategy_best_algorithm is None:
                strategy_best_algorithm = al_algorithm
                strategy_best_accuracy = test_accuracies[-1]

            print(f"{al_algorithm_idx+1}. {al_algorithm.upper()} summary - Test: {test_accuracies[-1]:.2f}% acc, Total training time: {format_time(training_time_total)}, Total AL time: {format_time(al_algorithm_time_total)}")
        print(f"Best AL-Algorithm for Budget Strategy {strategy}: {strategy_best_algorithm.upper()}, Test Accuracy: {strategy_best_accuracy:.2f}%")

        plt.figure()
        for al_algorithm in al_algorithms:
            train_val_set_size = simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["train_val_set_sizes"]
            test_accuracies = simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["test_accuracies"]
            plt.plot(train_val_set_sizes, test_accuracies, label=f"{al_algorithm.upper()}\nFinal: {test_accuracies[-1]:.2f}% acc",marker='o')
        plt.xlabel(f'Training Set Size ({(1 - TRAIN_VAL_RATIO) * 100:.0f}% for Validation)')
        plt.ylabel('Test Accuracy (%)')
        plt.title(f'Active Learning: Test Accuracy vs. Labelled Set Size\nSeed {seed} | Budget Strategy {strategy} | Best AL-Algorithm: {strategy_best_algorithm.upper()}')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    seed_time = time.time() - seed_time
    simulation_data["runtimes"]["seeds"][f"seed_{seed}"] = seed_time

simulation_time = time.time() - simulation_time
simulation_data["runtimes"]["simulation"] = simulation_time

# Plot average results for all AL algorithms for each budget strategy over all seeds
print("SIMULATION COMPLETE")
print(f"Total Runtime: {format_time(simulation_time)}")
if len(seeds) > 1:
    print("\nPlotting average AL-algorithm performances for each budget strategy across all seeds...")
    for strategy in selected_strategies:
        fig = plot_al_performance_across_seeds(
            simulation_data,
            strategy,
            al_algorithms
        )
        plt.show()
else:
    print("Only one seed, no need to plot average peformance across all seeds.")




# Map the present algorithms to their abbreviations
selected_abbreviations = [algorithm_abbreviations[algo] for algo in al_algorithms if algo in algorithm_abbreviations]

# Construct the filename
file_name = (
    f"model_{int(TRAIN_VAL_RATIO*100)}_{EPOCHS}_"
    f"seeds_{'-'.join(map(str, seeds))}_"
    f"baseline_{train_full_dataset_baseline}_"
    f"batch_{'-'.join(map(str, label_batch_sizes))}_"
    f"ratio_{RATIO_BATCH_TO_INIT_DATASET}_"
    f"algo_{'-'.join(selected_abbreviations)}_"
    f"iter_{NUM_TRAIN_AL_ITERATIONS}"
)

file_path = os.path.join(relative_save_folder, file_name)

# Ensure the folder exists
os.makedirs(relative_save_folder, exist_ok=True)

# Save the results to a JSON file if enabled and the file does not already exist
if save_results:
    if not os.path.exists(file_path): # Check if file already exists
        # Save the dictionary as a JSON file
        with open(file_path, "w") as file:
            json.dump(simulation_data, file, indent=4)
        print(f"\nData \n{file_name} \nsaved to \n{file_path}")
    else:
        print(f"\nFile \n{file_name} \nalready exists in \n{file_path}, \nskipping save.")
else:
    print("\nResults not saved, set save_results to True to save results.")
