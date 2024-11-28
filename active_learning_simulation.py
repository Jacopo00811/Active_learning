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

## Dataset & Model Parameters ##
dataset_name = "CIFAR-10" # Dataset to use for the simulation (only CIFAR-10 supported)
model_name = "ResNet-18" # Model to use for the simulation (only ResNet-18 supported)
pretrained_weights = True # Enable to use pretrained weights for the model

## Simulation Parameters ##
save_results = True # Enable to save results to file
relative_save_folder = "./run_results" # Define relative save folder

## Model Hyperparameters ##
TRAIN_VAL_RATIO = 0.8 # Ratio of training data to use for training (remaining is used for validation)
EPOCHS = 10 # Number of epochs to train the model for each budget size
BATCH_SIZE = 64 # Batch size for data loaders

## Seed and Baseline Parameters ##
seeds = [0, 1] # Set random seeds to decrease uncertainty
train_full_dataset_baseline = True # Enable to run the model with all labelled data to establish maximum performance baseline

## Active Learning Algorithm Parameters ##
AL_ALGORITHMS = {
    'random': {"active": True},
    'uncertainty': {"active": True},
    'typiclust': {"active": True},
    'margin': {"active": False}, # Example additional algorithm, not implemented yet
    'entropy': {"active": False} # Example additional algorithm, not implemented yet
}

## Budget Strategy Parameters ##
BUDGET_STRATEGIES = {
    1: {"active": True, "initial_size": 1000, "query_size": 250, "num_al_iterations": 10}, # Final size: 3500
    2: {"active": True, "initial_size": 4000, "query_size": 500, "num_al_iterations": 10}, # Final size: 9000
    3: {"active": False, "initial_size": 10000, "query_size": 1000, "num_al_iterations": 10}, # Final size: 20000
    4: {"active": False, "initial_size": 22000, "query_size": 2000, "num_al_iterations": 10}, # Final size: 42000 (almost full training/validation set size of 50000)
}

### End of Parameters ###


# Retrieve selected AL algorithms
al_algorithms = [algo for algo, config in AL_ALGORITHMS.items() if config["active"]]

# Setup budget strategies
budget_strategies = [num for num, config in BUDGET_STRATEGIES.items() if config["active"]]
budget_initial_sizes = [BUDGET_STRATEGIES[num]["initial_size"] for num in budget_strategies]
budget_query_sizes = [BUDGET_STRATEGIES[num]["query_size"] for num in budget_strategies]
budget_total_al_iterations = [BUDGET_STRATEGIES[num]["num_al_iterations"] for num in budget_strategies]
budget_final_sizes = [budget_initial_sizes[i] + budget_query_sizes[i] * budget_total_al_iterations[i] for i in range(len(budget_strategies))]

# Calculate total number of baseline and training/AL iterations (used for status printing)
num_seeds = len(seeds)
num_baselines = len(seeds) if train_full_dataset_baseline else 0
num_strategies = len(budget_strategies)
num_al_algorithms = len(al_algorithms)

# Set the device to CUDA / MPS if available, otherwise to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Store configuration data
simulation_data = {
    "config": {
        "simulation": {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "pretrained_weights": pretrained_weights,
            "device": str(device),
            "save_results": save_results,"relative_save_folder": relative_save_folder
        },
        "model": {
            "train_val_ratio": TRAIN_VAL_RATIO,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        },
        "seed_and_baseline": {
            "seeds": seeds,
            "train_full_dataset_baseline": train_full_dataset_baseline
        },
        "active_learning": {
            "al_algorithms": al_algorithms,
        },
        "budget": {
            "budget_strategies": budget_strategies,
            "budget_initial_sizes": budget_initial_sizes,
            "budget_query_sizes": budget_query_sizes,
            "budget_total_al_iterations": budget_total_al_iterations,
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
        for strategy in budget_strategies
    },
    "training_and_al_algorithms": {
        f"seed_{seed}_budget_{strategy}_{algo}": {"training": None, "al": None}
        for seed in seeds
        for strategy in budget_strategies
        for algo in al_algorithms
    }
}

# Initialize results data storage
simulation_data["results"] = {}


# Construct the filename
file_name = (
    f"{dataset_name.lower()}_"
    f"{model_name.lower()}_ptw{1 if pretrained_weights else 0}_"
    f"tvr{int(TRAIN_VAL_RATIO*100)}_"
    f"ep{EPOCHS}_"
    f"bs{BATCH_SIZE}_"
    f"sds{'-'.join(map(str, seeds))}_"
    f"bl{1 if train_full_dataset_baseline else 0}_"
    f"algo-{'-'.join([algo[:3] for algo in al_algorithms])}_"
    f"bsr{'-'.join(map(str, budget_strategies))}_"
    f"bis{'-'.join(map(str, budget_initial_sizes))}_"
    f"bqs{'-'.join(map(str, budget_query_sizes))}_"
    f"bni{'-'.join(map(str, budget_total_al_iterations))}"
)


file_path = os.path.join(relative_save_folder, file_name)

# Ensure the folder exists
os.makedirs(relative_save_folder, exist_ok=True)


# Print simulation configuration
print("============= ACTIVE LEARNING SIMULATION CONFIGURATION =============")

print("\nDATASET & MODEL:")
print(f"- Dataset: {dataset_name}")
print(f"- Model: {model_name}")
print(f"- Pretrained Weights Enabled: {pretrained_weights}")

print("\nSIMULATION PARAMETERS:")
print("- Running on device:", str(device).upper())
print(f"- Save Results Enabled: {save_results}")

print(f"\nMODEL HYPERPARAMETERS:")
print(f"- Training/Validation Split Ratio: {TRAIN_VAL_RATIO}")
print(f"- Epochs per Training Iteration: {EPOCHS}")
print(f"- Batch Size: {BATCH_SIZE}")

print(f"\nSEEDS & BASELINE:")
print(f"- Seeds: {seeds}")
print(f"- Train Full Dataset Baseline Enabled: {train_full_dataset_baseline}")

print(f"\nACTIVE LEARNING:")
print(f" - Active AL Algorithms: {', '.join(al_algorithms).upper()}")

print(f"\nBUDGET STRATEGIES:")
for i, strategy in enumerate(budget_strategies):
    print(f"  - Budget Strategy {strategy}:")
    print(f"    - Initial size: {budget_initial_sizes[i]} → Final size: {budget_final_sizes[i]}")
    print(f"    - Query size: {budget_query_sizes[i]} | AL Iterations: {budget_total_al_iterations[i]}")

print("""
EPOCHS BREAKDOWN (of varying dataset sizes):
Total Epochs: {total:,}
├── Baseline: {baseline:,}
└── Training: {training:,}""".format(
   total=EPOCHS * (num_baselines + num_strategies*num_al_algorithms*(sum(budget_total_al_iterations)+1)),
   baseline=EPOCHS * num_baselines,
   training=EPOCHS * num_strategies*num_al_algorithms*(sum(budget_total_al_iterations)+1)
))

print(f"""
SIMULATION STRUCTURE:
Seeds: {len(seeds)}
│
├── Baselines: {1 if train_full_dataset_baseline else 0}
│   └── Training
│       └── Epochs: {EPOCHS if train_full_dataset_baseline else 0}
│
└── Budget Strategies: {len(budget_strategies)}
    └── AL-Algorithms: {len(al_algorithms)}
        └── Iterations: {(np.mean(budget_total_al_iterations) + 1)} (on average)
            ├── AL (skips first iteration)
            └── Training
                └── Epochs: {EPOCHS}
""")

print("====================================================================")


# Save the results to a JSON file if enabled and the file does not already exist
if save_results:
    if os.path.exists(file_path): # Check if file already exists
        raise FileExistsError(f"Simulation {file_name} already exists in {file_path}\nSimulation with current config already has saved results from a previous run.\nIdentical config = Identical results, check saved simulation if data is needed.\nIf you want to run the simulation again, change the configuration, delete the existing file, or alternatively set save_results to False.")


# Load the dataset
if dataset_name == "CIFAR-10":
    # Load CIFAR-10 with transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 Dataset (Training and Test splits)
    train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
else:
    raise ValueError(f"Dataset {dataset_name} not supported.")

# Split into training and validation sets
train_size = int(TRAIN_VAL_RATIO * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

print("\nBeginning AL Simulation...")
print(f"Save results: {save_results}")
print(f"Seeds to run: {seeds}")
print(f"Baselines per seed: {1 if train_full_dataset_baseline else 0}")
print(f"Budget strategies per seed: {budget_strategies}")
print(f"AL-algorithms per budget strategy: {al_algorithms}")

# Initialize timer
simulation_time = time.time()

# Run the active learning evaluation loop for each seed
for seed_idx, seed in enumerate(seeds):
    seed_time = time.time()

    print(f"\n\n[SEED {seed} ({seed_idx+1}/{num_seeds})]")
    
    # Setup model
    if model_name == "ResNet-18":
        # Load ResNet-18 model with or without pretrained weights
        if pretrained_weights:
            # Load pretrained weights with metadata
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        else:
            weights = None
            transform = None

        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)

    else:
        raise ValueError(f"Model {model_name} not supported.")


    # Set random seed and generator used for data loaders
    generator = set_global_seed(seed)
    simulation_data["results"][f"seed_{seed}"] = {}

    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    # Data loaders
    full_train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=generator)
    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=generator) # Only used for training baseline with all labels
    full_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, generator=generator) # Only used for validation baseline with all labels
    full_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, generator=generator) # Both used for testing with all labels and for Train/AL iterations


    # (If enabled) Train the model with entire dataset labelled to establish maximum performance baseline
    if train_full_dataset_baseline:
        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds}) | TRAINING FULL DATASET BASELINE]")

        full_dataset_baseline_time = time.time()

        reset_model_weights(model)

        train_model(
            device=device, 
            model=model, 
            epochs=EPOCHS, 
            train_loader=full_train_loader, 
            val_loader=full_val_loader, 
        )

        test_accuracy = evaluate_model(
            device=device, 
            model=model, 
            data_loader=full_test_loader
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
    for strategy_idx, strategy in enumerate(budget_strategies):
        budget_strategy_time = time.time()

        budget_initial_size = budget_initial_sizes[strategy_idx]
        budget_query_size = budget_query_sizes[strategy_idx]
        budget_al_iterations = budget_total_al_iterations[strategy_idx]
        budget_final_size = budget_final_sizes[strategy_idx]

        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds})) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies})]")
        print(f"Initial size: {budget_initial_size} → Final size: {budget_final_size} | Query size: {budget_query_size} | AL Iterations: {budget_al_iterations}")

        simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"] = {}

        for al_algorithm_idx, al_algorithm in enumerate(al_algorithms):
            print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds}) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies}) | AL: {al_algorithm.upper()} ({al_algorithm_idx+1}/{num_al_algorithms})]")

            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm] = {}
            
            # Run active learning loop, return training set sizes and corresponding test accuracies
            train_val_set_sizes, test_accuracies, training_time_total, al_time_total = active_learning_loop(
                device=device, 
                model=model,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                train_val_ratio=TRAIN_VAL_RATIO,  
                train_val_dataset=train_val_dataset,
                full_test_loader=full_test_loader, 
                generator=generator,
                budget_initial_size=budget_initial_size,
                budget_query_size=budget_query_size,
                budget_al_iterations=budget_al_iterations,
                al_algorithm=al_algorithm
            )

            # Store results
            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["train_val_set_sizes"] = train_val_set_sizes
            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["test_accuracies"] = test_accuracies

            # Store training and AL algorithm runtimes
            simulation_data["runtimes"]["training_and_al_algorithms"][f"seed_{seed}_budget_{strategy}_{al_algorithm}"]["training"] = training_time_total
            simulation_data["runtimes"]["training_and_al_algorithms"][f"seed_{seed}_budget_{strategy}_{al_algorithm}"]["al"] = al_time_total

            print(f"{al_algorithm.upper()} complete - Test: {test_accuracies[-1]:.2f}% acc, Total training time: {format_time(training_time_total)}, Total AL time: {format_time(al_time_total)}")


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

            print(f"{al_algorithm_idx+1}. {al_algorithm.upper()} summary - Test: {test_accuracies[-1]:.2f}% acc, Total training time: {format_time(training_time_total)}, Total AL time: {format_time(al_time_total)}")
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
    for strategy in budget_strategies:
        fig = plot_al_performance_across_seeds(
            simulation_data,
            strategy,
            al_algorithms
        )
        plt.show()
else:
    print("\nSkipping average AL-algorithm performance plots - Only one seed was run.")


# Save the results to a JSON file if enabled and the file does not already exist
if save_results:
    if not os.path.exists(file_path): # Check if file already exists
        # Save the dictionary as a JSON file
        with open(file_path, "w") as file:
            json.dump(simulation_data, file, indent=4)
        print(f"\nSimulation results saved to \n{file_path}")
else:
    print("\nSAVING DISABLED - Results not saved to file. Set save_results to True to save results.")
