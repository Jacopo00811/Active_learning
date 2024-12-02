import json
import os
import matplotlib.pyplot as plt
from util_functions import format_time
from analysis_functions import plot_al_performance_across_seeds
import numpy as np


## TESTING RESULTS ##
# With pretrained weights
    # With baseline
        # Baseline Complete - Test: 77.99% acc
        # RANDOM summary - Test: 60.70% acc
        # UNCERTAINTY summary - Test: 64.25% acc
    # Without baseline
        # RANDOM summary - Test: 60.70% acc
        # UNCERTAINTY summary - Test: 64.25% acc

# Without pretrained weights
    # With baseline
        # Baseline Complete - Test: 70.48% acc
        # RANDOM summary - Test: 42.45% acc
        # UNCERTAINTY summary - Test: 41.95% acc
    # Without baseline
        # RANDOM summary - Test: 42.45% acc
        # UNCERTAINTY summary - Test: 41.95% acc

# Observations:
# - Baseline vs no baseline: Doesn't affect AL performance results, indicating model weights re-initialization is working correctly (no weights leakage)
# - Pretrained vs not pretrained: Baseline performance is slightly better with pretrained weights, AL performance is significantly better with pretrained weights. Indicates that pretrained weights are beneficial for AL performance, and that they are no longer reset to random weights during AL iterations.



def reconstruct_rel_file_path_from_config(relative_save_folder, dataset_name, model_name, pretrained_weights, train_val_ratio, epochs, batch_size, seeds, train_full_dataset_baseline, al_algorithms, budget_strategies, budget_initial_sizes, budget_query_sizes, budget_total_al_iterations):
    file_name = (
        f"{dataset_name.lower()}_"
        f"{model_name.lower()}_"
        f"ptw{1 if pretrained_weights else 0}_"
        f"tvr{int(train_val_ratio*100)}_"
        f"ep{epochs}_"
        f"bs{batch_size}_"
        f"sds{'-'.join(map(str, seeds))}_"
        f"bl{1 if train_full_dataset_baseline else 0}_"
        f"algo-{'-'.join([algo[:3] for algo in al_algorithms])}_"
        f"bsr{'-'.join(map(str, budget_strategies))}_"
        f"bis{'-'.join(map(str, budget_initial_sizes))}_"
        f"bqs{'-'.join(map(str, budget_query_sizes))}_"
        f"bni{'-'.join(map(str, budget_total_al_iterations))}"
    )
    print(f"Reconstructed relative file path:\n{file_name}")

    file_path = os.path.join(relative_save_folder, file_name)
    return file_path



def load_simulation_data(file_path):
    try:
        with open(file_path, "r") as file:
            simulation_data = json.load(file)
        print(f"Successfully loaded data from {file_path}")
        return simulation_data
    except FileNotFoundError:
        print(f"No file found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return None


### DEFINE CONFIGURATION VARIABLES FOR FILE PATH RECONSTRUCTION ###
dataset_name = "CIFAR-10"
model_name = "ResNet-18"
pretrained_weights = True
relative_save_folder = "./run_results"
TRAIN_VAL_RATIO = 0.8
EPOCHS = 8
BATCH_SIZE = 32
seeds = [0, 1, 2, 3, 4]
train_full_dataset_baseline = True
AL_ALGORITHMS = {
    'random': {"active": True},
    'uncertainty': {"active": True},
    'typiclust': {"active": True},
    'margin': {"active": True}, 
    'entropy': {"active": True},
    'badge': {"active": True},
    'coreset': {"active": True},
}
BUDGET_STRATEGIES = {
    1: {"active": True, "initial_size": 1000, "query_size": 250, "num_al_iterations": 10},
    2: {"active": True, "initial_size": 4000, "query_size": 500, "num_al_iterations": 10},
    3: {"active": True, "initial_size": 10000, "query_size": 1000, "num_al_iterations": 10},
    4: {"active": False, "initial_size": 22000, "query_size": 2000, "num_al_iterations": 10},
}
# Calculated from the above configuration
al_algorithms = [algo for algo, config in AL_ALGORITHMS.items() if config["active"]]
budget_strategies = [num for num, config in BUDGET_STRATEGIES.items() if config["active"]]
budget_initial_sizes = [BUDGET_STRATEGIES[num]["initial_size"] for num in budget_strategies]
budget_query_sizes = [BUDGET_STRATEGIES[num]["query_size"] for num in budget_strategies]
budget_total_al_iterations = [BUDGET_STRATEGIES[num]["num_al_iterations"] for num in budget_strategies]
budget_final_sizes = [budget_initial_sizes[i] + budget_query_sizes[i] * budget_total_al_iterations[i] for i in range(len(budget_strategies))]


# Reconstruct the file path from the configuration variables
file_path = reconstruct_rel_file_path_from_config(
    relative_save_folder=relative_save_folder,
    dataset_name=dataset_name, 
    model_name=model_name, 
    pretrained_weights=pretrained_weights, 
    train_val_ratio=TRAIN_VAL_RATIO, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    seeds=seeds, 
    train_full_dataset_baseline=train_full_dataset_baseline, 
    al_algorithms=al_algorithms, 
    budget_strategies=budget_strategies,
    budget_initial_sizes=budget_initial_sizes, 
    budget_query_sizes=budget_query_sizes, 
    budget_total_al_iterations=budget_total_al_iterations
)

# Load the simulation data from the file path
simulation_data = load_simulation_data(file_path)
if simulation_data is not None:
    print(f"Simulation data loaded successfully.\nSimulation {file_path}")
    config = simulation_data["config"]
    runtimes = simulation_data["runtimes"]
    results = simulation_data["results"]
else:
    print("No simulation data loaded")

"""
print("\n\n\nAnalyzing simulation data...")
simulation_time = runtimes["simulation"]
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
    print("Only one seed, no need to plot average peformance across all seeds.")
"""





def average_al_results(results, al_algorithms, budget_strategies, seeds):
    averaged_results = {}
    
    for algo in al_algorithms:
        # Initialize storage for final algorithm results
        algo_results = {
            'mean_train_val_sizes': [],
            'mean_test_accuracies': [],
            'std_test_accuracies': []
        }

        mean_train_val_sizes = []
        mean_test_accuracies = []
        std_test_accuracies = []

        # Store intermediate results for each budget strategy
        strategy_level_results = []
        
        # First, process each budget strategy separately
        for strategy in budget_strategies:
            # For this strategy, collect and average across all seeds
            seed_train_val_sizes = []
            seed_test_accuracies = []
            
            # Gather results from all seeds for current strategy
            for seed in seeds:
                seed_name = f"seed_{seed}"
                train_val_set_sizes = simulation_data["results"][seed_name][f"budget_strategy_{strategy}"][algo]["train_val_set_sizes"]
                test_accuracies = simulation_data["results"][seed_name][f"budget_strategy_{strategy}"][algo]["test_accuracies"]
                seed_train_val_sizes.append(train_val_set_sizes)
                seed_test_accuracies.append(test_accuracies)
            
            # Convert to numpy arrays for easier computation
            strategy_mean_sizes = np.mean(seed_train_val_sizes, axis=0)
            strategy_mean_accuracies = np.mean(seed_test_accuracies, axis=0)
            strategy_std_accuracies = np.std(seed_test_accuracies, axis=0)
            print(strategy_mean_sizes)
            print(strategy_mean_accuracies)
            print(strategy_std_accuracies)
            
            mean_train_val_sizes.append(strategy_mean_sizes)
            mean_test_accuracies.append(strategy_mean_accuracies)
            std_test_accuracies.append(strategy_std_accuracies)
        
        mean_train_val_sizes = np.concatenate(mean_train_val_sizes, axis=0)
        mean_test_accuracies = np.concatenate(mean_test_accuracies, axis=0)
        std_test_accuracies = np.concatenate(std_test_accuracies, axis=0)

        print(mean_train_val_sizes)
        print(mean_test_accuracies)
        print(std_test_accuracies)
        print(algo, "\n\n")
        
        # Calculate final statistics across strategies
        algo_results['mean_train_val_sizes'] = mean_train_val_sizes
        algo_results['mean_test_accuracies'] = mean_test_accuracies
        # Combine standard deviations appropriately
        algo_results['std_test_accuracies'] = std_test_accuracies
        
        averaged_results[algo] = algo_results
    
    if simulation_data["config"]["seed_and_baseline"]["train_full_dataset_baseline"]:
        baseline_all_accuracies = []
        for seed in seeds:
            seed_name = f"seed_{seed}"
            baseline_test_accuracy = results["results"][seed_name]["full_dataset_baseline"]
            baseline_all_accuracies.append(baseline_test_accuracy)
        
        baseline_mean_accuracy = np.mean(baseline_all_accuracies)
        baseline_std_accuracies = np.std(baseline_all_accuracies)

        averaged_results["baseline"] = {
            'baseline_mean_accuracy': baseline_mean_accuracy,
            'baseline_std_accuracy': baseline_std_accuracies
        }

    return averaged_results

# Example usage:

averaged_results = average_al_results(simulation_data, al_algorithms, budget_strategies, seeds)

# Plot with confidence intervals:
def plot_sim_results(averaged_results, al_algorithms, BUDGET_STRATEGIES, simulation_data, enable_std=True):
    plt.figure()
    for algo in al_algorithms:
        if algo == "random":
            marker='o'
        else:
            marker='x'
            

        algo_performance = averaged_results[algo]
        mean_acc = algo_performance['mean_test_accuracies']
        std_acc = algo_performance['std_test_accuracies']
        mean_sizes = algo_performance['mean_train_val_sizes']
        plt.plot(mean_sizes, mean_acc, marker=marker, label=algo, markersize=6, linewidth=1)
        if enable_std:
            plt.fill_between(mean_sizes, 
                        mean_acc - 2 * std_acc,
                        mean_acc + 2 * std_acc,
                        alpha=0.3)

    # Add vertical lines for initial sizes
    for strategy_id, strategy_config in BUDGET_STRATEGIES.items():
        initial_size = strategy_config['initial_size'] 
        if strategy_config['active']:
            # Adding a vertical line with a light gray color and dashed style
            plt.axvline(x=initial_size, color='gray', linestyle='--', alpha=0.5)
        # Optionally add a text label above the line
        # plt.text(initial_size, plt.ylim()[1], f'Initial Size {initial_size}', rotation=90, va='bottom', ha='right')    

    if simulation_data["config"]["seed_and_baseline"]["train_full_dataset_baseline"]:
        baseline_mean_accuracy = averaged_results["baseline"]["baseline_mean_accuracy"]
        plt.axhline(y=baseline_mean_accuracy, color='red', linestyle='-', alpha=0.5)

    plt.legend()
    plt.show()

plot_sim_results(averaged_results, al_algorithms, BUDGET_STRATEGIES, simulation_data, enable_std=False)



"""
POTENTIAL ANALYSIS TOOLS WE COULD IMPLEMENT:

1. Learning Efficiency Analysis:
   - Calculates how much more efficient an AL algorithm is compared to random sampling
   - Uses Area Under the Learning Curve (AUC) to quantify overall performance
   - Reports percentage improvement over random baseline

2. Label Diversity Analysis:
   - Measures how diverse the selected samples are using entropy
   - Helps identify if the algorithm is focusing too much on certain classes/regions
   - Important for detecting potential sampling biases

3. Label Efficiency Metrics:
   - Calculates how many labels are needed to reach a target accuracy
   - Useful for budget planning and comparing algorithms' practical efficiency
   - Includes variance across seeds for reliability assessment

4. Learning Rate Analysis:
   - Plots the rate of improvement (derivative of learning curve)
   - Shows where each algorithm learns most efficiently
   - Helps identify diminishing returns

5. Comprehensive Summary Report:
   - Combines all metrics into a single DataFrame
   - Includes confidence intervals from multiple seeds
   - Easy to compare algorithms across multiple dimensions

Additional suggestions for evaluation:

1. Statistical Tests:
   - Add significance tests between algorithms
   - Could use paired t-tests at key points in the learning curve
   - Test if differences in final performance are statistically significant

2. Cost-Benefit Analysis:
   - If different types of samples have different labeling costs
   - Track cumulative annotation cost instead of just sample count
   - Calculate return on investment for each algorithm

3. Robustness Analysis:
   - Test with different initial seed sizes
   - Evaluate sensitivity to batch size
   - Measure stability across different random seeds

4. Confusion Matrix Analysis:
   - Track how confusion matrices evolve
   - Identify which classes benefit most from active learning
   - Detect if certain classes are being neglected
"""



