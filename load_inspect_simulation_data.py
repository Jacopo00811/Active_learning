import json
import os
import matplotlib.pyplot as plt
from util_functions import format_time
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd


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



def average_strategy_results(results, al_algorithms, budget_strategies, seeds):
    averaged_results = {}
    
    baseline_mean_accuracy = []
    for seed in seeds:
        baseline_accuracy = results['results'][f'seed_{seed}']["full_dataset_baseline"]
        baseline_mean_accuracy.append(baseline_accuracy)

    averaged_results['baseline'] = np.mean(baseline_mean_accuracy)

    for strategy in [strategy for strategy, config in budget_strategies.items() if config['active']]:

        train_val_set_sizes = results['results'][f'seed_{seeds[0]}'][f"budget_strategy_{strategy}"]['random']['train_val_set_sizes']
        averaged_results[f"budget_strategy_{strategy}"] = {
            'train_val_set_sizes': train_val_set_sizes
        }

        for al_algorithm in al_algorithms:
            # For this strategy, collect and average across all seeds
            seed_test_accuracies = []
            
            # Gather results from all seeds for current strategy
            for seed in seeds:
                seed_name = f"seed_{seed}"
                test_accuracies = simulation_data["results"][seed_name][f"budget_strategy_{strategy}"][al_algorithm]["test_accuracies"]
                seed_test_accuracies.append(test_accuracies)
            
            # Convert to numpy arrays for easier computation
            mean_accuracies = np.mean(seed_test_accuracies, axis=0)
            std_accuracies = np.std(seed_test_accuracies, axis=0)
            
            averaged_results[f"budget_strategy_{strategy}"][al_algorithm] = {
                'mean_accuracies': mean_accuracies,
                'std_accuracies': std_accuracies
            }
    return averaged_results

def al_improvements_over_random(averaged_results):
    # Get list of strategies and algorithms to compare
    strategies = [strategy for strategy in averaged_results if strategy != 'baseline']
    algorithms_to_compare = [algo for algo in al_algorithms if algo != 'random']

    # Create DataFrame with strategies as index and algorithms as columns
    df = pd.DataFrame(index=strategies, columns=algorithms_to_compare)

    # For each strategy
    for strategy in strategies:

        # Get random accuracies for this strategy
        random_mean_accuracies = averaged_results[strategy]['random']['mean_accuracies']
        
        # For each algorithm we want to compare
        for algo in algorithms_to_compare:
            # Store improvements for this algorithm
            improvements = []
            
            # Calculate improvement for each iteration
            for i, random_accuracy in enumerate(random_mean_accuracies):
                if i == 0:
                    continue

                algo_accuracy = averaged_results[strategy][algo]['mean_accuracies'][i]
                improvement = ((algo_accuracy - random_accuracy) / random_accuracy) * 100
                improvements.append(improvement)

                if i == 0:
                    print(random_accuracy, algo_accuracy)
            
            # Calculate mean improvement and assign to DataFrame
            # Use .loc to explicitly locate the cell we want to update
            df.loc[strategy, algo] = f"{np.mean(improvements):.2f}%"

    return df

def plot_sim_results(averaged_results, al_algorithms, strategies, enable_std=True, enable_baseline=True, enable_initial_lines=True):

    plt.figure()

    algo_colour_map = {
        'random': "black",
        'uncertainty': "blue",
        'typiclust': "green",
        'margin': "purple",
        'entropy': "orange",
        'badge': "brown",
        'coreset': "pink",
    }

    labelled_algos = set()

    for strategy in strategies:

        strategy_sizes = averaged_results[strategy]['train_val_set_sizes']

        for algo in al_algorithms:
            algo_mean_accuracies = averaged_results[strategy][algo]['mean_accuracies']
            algo_std_accuracies = averaged_results[strategy][algo]['std_accuracies']
            
            if algo in labelled_algos:
                label = None
            else:
                label = algo
                labelled_algos.add(algo)

            plt.plot(strategy_sizes, algo_mean_accuracies, marker='o', label=label, color=algo_colour_map[algo] ,markersize=6, linewidth=1, alpha=1)
            if enable_std:
                plt.fill_between(strategy_sizes, 
                            algo_mean_accuracies - 2 * algo_std_accuracies,
                            algo_mean_accuracies + 2 * algo_std_accuracies,
                            alpha=0.3)

        # Adding a vertical line with a light gray color and dashed style
        if enable_initial_lines:
            plt.axvline(x=strategy_sizes[0], color='black', linestyle='--', alpha=0.9)

    if enable_baseline:
        plt.axhline(y=averaged_results['baseline'], color='red', label='Full Dataset Baseline', linestyle='-', linewidth=3, alpha=1)

    plt.title(f"{strategies}")
    plt.legend()
    plt.show()


averaged_results = average_strategy_results(simulation_data, AL_ALGORITHMS, BUDGET_STRATEGIES, seeds)

df = al_improvements_over_random(averaged_results)
print(df)

# Plot with confidence intervals:
strategies = [strategy for strategy in averaged_results if strategy != 'baseline']
plot_sim_results(averaged_results, al_algorithms, strategies, enable_std=False)

# Plot each strategy separately
plot_sim_results(averaged_results, al_algorithms, [strategies[0]], enable_std=False)
plot_sim_results(averaged_results, al_algorithms, [strategies[1]], enable_std=False)
plot_sim_results(averaged_results, al_algorithms, [strategies[2]], enable_std=False)

# Plot without baseline but with std
plot_sim_results(averaged_results, al_algorithms, [strategies[0]], enable_std=True, enable_baseline=False, enable_initial_lines=False)
plot_sim_results(averaged_results, al_algorithms, [strategies[1]], enable_std=True, enable_baseline=False, enable_initial_lines=False)
plot_sim_results(averaged_results, al_algorithms, [strategies[2]], enable_std=True, enable_baseline=False, enable_initial_lines=False)


# # plot_sim_results(averaged_results, al_algorithms, BUDGET_STRATEGIES, simulation_data, enable_std=False)
# print("|||||||||")
# print(averaged_results["random"].keys())
# print(averaged_results["random"]["mean_test_accuracies"])
# algorithms = list(averaged_results.keys())[:-1]

# heat_map = np.zeros((len(algorithms), len(algorithms)))
# for i, algo1 in enumerate(algorithms):
#     for j, algo2 in enumerate(algorithms):
#         if i != j:
#             accuracies1 = averaged_results[algo1]['mean_test_accuracies'][:10]
#             print(algo2)
#             accuracies2 = averaged_results[algo2]['mean_test_accuracies'][:10]
#             t_stat, p_value = ttest_ind(accuracies1, accuracies2)
#             heat_map[i, j] = p_value
#         else:
#             heat_map[i, j] = 1  # Diagonal should be 1 (no difference)
# print(heat_map)

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



