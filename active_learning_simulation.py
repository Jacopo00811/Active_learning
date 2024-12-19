import os
import json
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from util_functions import set_global_seed, full_training_set_baseline, active_learning_loop, format_time, create_imbalanced_cifar10
from analysis_functions import plot_al_performance_across_seeds
import numpy as np
from torch.utils.data import Subset



### Begin of Parameters ###

## Dataset & Model Parameters ##
dataset_name = "CIFAR-10" # "CIFAR-10" or "MNIST"
imbalanced_dataset = False # Enable to create an imbalanced dataset (only for CIFAR-10)
model_name = "ResNet-18" # Model to use for the simulation (only ResNet-18 supported)

## Simulation Parameters ##
save_results = True # Enable to save results to file
relative_save_folder = "./run_results" # Define relative save folder

## Model Hyperparameters ##
pretrained_weights = True # Enable to use pretrained weights for the model
TRAIN_VAL_RATIO = 0.8 # Ratio of training data to use for training (remaining is used for validation)
EPOCHS = 8 # Number of epochs to train the model for each budget size
BATCH_SIZE = 32 # Batch size for data loaders

## Seed and Baseline Parameters ##
seeds = [0, 1, 2, 3, 4] # Set random seeds to decrease uncertainty
train_full_dataset_baseline = True # Enable to run the model with all labelled data to establish maximum performance baseline

## Active Learning Algorithm Parameters ##
AL_ALGORITHMS = {
    'random': {"active": True},
    'uncertainty': {"active": True},
    'typiclust': {"active": True},
    'margin': {"active": True}, 
    'entropy': {"active": True},
    'badge': {"active": True},
    'coreset': {"active": True},
}

## Budget Strategy Parameters ##
BUDGET_STRATEGIES = {
    1: {"active": True, "initial_size": 1000, "query_size": 250, "num_al_iterations": 10}, # Final size: 3500
    2: {"active": True, "initial_size": 4000, "query_size": 500, "num_al_iterations": 10}, # Final size: 9000
    3: {"active": False, "initial_size": 10000, "query_size": 1000, "num_al_iterations": 10}, # Final size: 20000
    4: {"active": False, "initial_size": 22000, "query_size": 2000, "num_al_iterations": 10}, # Final size: 42000 (almost full training/validation set size of 50000)
}

# Mimic real-world frequency distribution for CIFAR-10 classes (if enabled)
base_ratios = {
    0: 1.0,    # common classes
    1: 1.0,
    2: 0.7,    # moderately common
    3: 0.7,
    4: 0.4,    # uncommon
    5: 0.4,
    6: 0.2,    # rare
    7: 0.2,
    8: 0.1,    # very rare
    9: 0.1
}

### End of Parameters ###



## Device Selection ##
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # Set the device to CUDA / MPS if available, otherwise to CPU.

## AL Algorithms Setup ##
al_algorithms = [algo for algo, config in AL_ALGORITHMS.items() if config["active"]] # Retrieve selected AL algorithms

## Budget Strategies Setup ##
budget_strategies = [num for num, config in BUDGET_STRATEGIES.items() if config["active"]] # Retrieve selected budget strategies
budget_initial_sizes = [BUDGET_STRATEGIES[num]["initial_size"] for num in budget_strategies] # Retrieve initial sizes for each budget strategy
budget_query_sizes = [BUDGET_STRATEGIES[num]["query_size"] for num in budget_strategies] # Retrieve query sizes for each budget strategy
budget_total_al_iterations = [BUDGET_STRATEGIES[num]["num_al_iterations"] for num in budget_strategies] # Retrieve total number of AL iterations for each budget strategy
budget_final_sizes = [budget_initial_sizes[i] + budget_query_sizes[i] * budget_total_al_iterations[i] for i in range(len(budget_strategies))] # Calculate final sizes for each budget strategy

# Check for overlap between budget strategies
for i in range(len(budget_strategies)):
    for j in range(i+1, len(budget_strategies)):
        budget_final_sizes[i] <= budget_initial_sizes[j], f"Overlap between budget strategies detected. Strategy {budget_strategies[i]} has final size {budget_final_sizes[i]} and strategy {budget_strategies[j]} has initial size {budget_initial_sizes[j]}."

## Simulation Saved Data Structure ##
# Store configuration data
simulation_data = {
    "config": {
        "simulation": {
            "save_results": save_results,"relative_save_folder": relative_save_folder,
            "device": str(device),
            "dataset_name": dataset_name,
            "model_name": model_name,
        },
        "model": {
            "pretrained_weights": pretrained_weights,
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

## File Name Generation & Save Folder Creation ##
file_name = ( # Construct the filename
    f"{dataset_name.lower()}_"
    f"{model_name.lower()}_"
    f"ptw{1 if pretrained_weights else 0}_"
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

file_path = os.path.join(relative_save_folder, file_name) # Construct the file path

os.makedirs(relative_save_folder, exist_ok=True) # Ensure the folder exists

## Size Parameters used for Status Printing ##
num_seeds = len(seeds)
num_baselines = len(seeds) if train_full_dataset_baseline else 0
num_strategies = len(budget_strategies)
num_al_algorithms = len(al_algorithms)

## Print Simulation Configuration ##
print("============= ACTIVE LEARNING SIMULATION CONFIGURATION =============")

print("\nDATASET & MODEL:")
print(f"- Dataset: {dataset_name}")
print(f"- Model: {model_name}")

print("\nSIMULATION PARAMETERS:")
print("- Running on device:", str(device).upper())
print(f"- Save Results Enabled: {save_results}")

print(f"\nMODEL HYPERPARAMETERS:")
print(f"- Pretrained Weights Enabled: {pretrained_weights}")
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


## Check if both the file already exists and saving is enabled ##
if save_results:
    if os.path.exists(file_path): # Check if file already exists
        raise FileExistsError(f"Simulation {file_name} already exists in {file_path}\nSimulation with current config already has saved results from a previous run.\nIdentical config = Identical results, check saved simulation if data is needed.\nIf you want to run the simulation again, change the configuration, delete the existing file, or alternatively set save_results to False.")

## Load the dataset and split into training and validation sets ##
if dataset_name == "CIFAR-10":
    # Load CIFAR-10 with transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 Dataset (Training and Test splits)
    train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a copy by loading the dataset again
    OG_train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

elif dataset_name == "MNIST":
    # Load MNIST with transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST specific normalization values
    ])

    # MNIST Dataset (Training and Test splits)
    train_val_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create a copy by loading the dataset again
    OG_train_val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

else:
    raise ValueError(f"Dataset {dataset_name} not supported.")


## Print Configuration Summary ##
print(f"\nConfig summary:")
print(f"- Save results: {save_results}")
print(f"- Device: {device} | Dataset: {dataset_name}")
print(f"- Model: {model_name} | Pretrained: {pretrained_weights} | Train/Val Ratio: {TRAIN_VAL_RATIO} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
print(f"- Seeds to run: {seeds}")
print(f"- Run full dataset baseline per seed: {train_full_dataset_baseline}")
print(f"- Budget strategies per seed: {budget_strategies}")
print(f"- AL-algorithms per budget strategy: {al_algorithms}")

print("\nBeginning AL Simulation...")


## Simulation Loop ##
simulation_time = time.time() # Initialize timer

# Run the active learning evaluation loop for each seed
for seed_idx, seed in enumerate(seeds):
    seed_time = time.time()
    simulation_data["results"][f"seed_{seed}"] = {}
    print(f"\n\n[SEED {seed} ({seed_idx+1}/{num_seeds})]")
    
    # Set random seed and generator used for data loaders
    initial_generator = set_global_seed(seed)

    # Create imbalanced CIFAR-10 dataset
    if imbalanced_dataset and dataset_name == "CIFAR-10":
        # Add small random variations to make it more realistic
        keep_ratios = {k: min(1.0, max(0.05, v + np.random.normal(0, 0.05))) 
                for k, v in base_ratios.items()}
        
        train_val_dataset = create_imbalanced_cifar10(OG_train_val_dataset, keep_ratios)
        print("Imbalanced CIFAR-10 training / validation created, len(train_val_dataset):", len(train_val_dataset))
        print(f"Test dataset is unchanged, len(test_dataset): {len(test_dataset)}")
    else:
        print("Imbalanced dataset disabled, skipping...")

    # Split into training and validation sets
    train_size = int(TRAIN_VAL_RATIO * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=initial_generator)

    # (If enabled) Train the model with entire dataset labelled to establish maximum performance baseline
    if train_full_dataset_baseline:
        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds}) | TRAINING FULL DATASET BASELINE]")

        full_dataset_baseline_time = time.time()

        test_accuracy = full_training_set_baseline(
            device=device, 
            dataset_name=dataset_name,
            model_name=model_name, 
            pretrained_weights=pretrained_weights, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            test_dataset=test_dataset, 
            generator=initial_generator)

        full_dataset_baseline_time = time.time() - full_dataset_baseline_time

        simulation_data["results"][f"seed_{seed}"]["full_dataset_baseline"] = test_accuracy
        simulation_data["runtimes"]["full_dataset_baselines"][f"seed_{seed}_full_dataset_baseline"] = full_dataset_baseline_time

        print(f"Baseline Complete - Test: {test_accuracy:.2f}% acc, Total time: {format_time(full_dataset_baseline_time)}")
    else:
        simulation_data["results"][f"seed_{seed}"]["full_dataset_baseline"] = None
        simulation_data["runtimes"]["full_dataset_baselines"][f"seed_{seed}_full_dataset_baseline"] = None
        print("Full dataset baseline disabled, skipping...")


    # For each budget strategy, train, evaluate, visualize and compare all AL strategies
    for strategy_idx, strategy in enumerate(budget_strategies):
        budget_strategy_time = time.time()
        simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"] = {}

        budget_initial_size = budget_initial_sizes[strategy_idx]
        budget_query_size = budget_query_sizes[strategy_idx]
        budget_al_iterations = budget_total_al_iterations[strategy_idx]
        budget_final_size = budget_final_sizes[strategy_idx]

        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds})) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies})]")
        print(f"Initial size: {budget_initial_size} → Final size: {budget_final_size} | Query size: {budget_query_size} | AL Iterations: {budget_al_iterations}")
        
        # Debug lists for initial test accuracy, should be the same for all AL algorithms since no AL happens in the first (0'th) iteration, only training on the same initial labelled set
        debug_initial_test_accuracy = []

        for al_algorithm_idx, al_algorithm in enumerate(al_algorithms):
            simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm] = {}
            
            print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds}) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies}) | AL: {al_algorithm.upper()} ({al_algorithm_idx+1}/{num_al_algorithms})]")
            
            # Run active learning loop, return training set sizes and corresponding test accuracies
            train_val_set_sizes, test_accuracies, training_time_total, al_time_total = active_learning_loop(
                device=device, 
                dataset_name=dataset_name,
                model_name=model_name,
                pretrained_weights=pretrained_weights,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                train_val_ratio=TRAIN_VAL_RATIO,  
                train_val_dataset=train_val_dataset,
                test_dataset=test_dataset,
                generator=initial_generator,
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

            # Debug: Store initial test accuracy for comparison
            debug_initial_test_accuracy.append(test_accuracies[0])
            if len(debug_initial_test_accuracy) > 1:
                assert abs(debug_initial_test_accuracy[-1] - debug_initial_test_accuracy[-2]) < 1e-6, "Initial test accuracy should be the same for all AL algorithms, as no AL happens in the first iteration and model is trained on the same initial labelled set."


        budget_strategy_time = time.time() - budget_strategy_time
        simulation_data["runtimes"]["budget_strategies"][f"seed_{seed}_budget_{strategy}"] = budget_strategy_time

        strategy_best_algorithm = None
        strategy_best_accuracy = 0

        print(f"\n[SEED {seed} ({seed_idx+1}/{num_seeds})) | BUDGET STRATEGY {strategy} ({strategy_idx+1}/{num_strategies}) | COMPLETED]")
        print(f"Total Runtime: {format_time(budget_strategy_time)}, Summary:")
        for al_algorithm_idx, al_algorithm in enumerate(al_algorithms):
            test_accuracies = simulation_data["results"][f"seed_{seed}"][f"budget_strategy_{strategy}"][al_algorithm]["test_accuracies"]
            training_time_total = simulation_data["runtimes"]["training_and_al_algorithms"][f"seed_{seed}_budget_{strategy}_{al_algorithm}"]["training"]
            al_time_total = simulation_data["runtimes"]["training_and_al_algorithms"][f"seed_{seed}_budget_{strategy}_{al_algorithm}"]["al"]

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
