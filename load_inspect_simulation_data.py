import json
import os
import matplotlib.pyplot as plt
from util_functions import format_time, plot_al_performance_across_seeds

"""
Personal:
- Change ALL data loaders to use remainders as well
- GPU accelerate uncertainty algorithm, check that it makes a difference
- Add tests to check if model weights are reset between iterations
- Add tests in active_learning_loop to ensure no data leakage between labelled training and validaiton subsets and unlabelled subset
- Change AL selection to same format as budget strategy selection, also easy to add abbreviation into it.
- Move initial/batch ratio inside of budget strategy selection, allows us to set custom initial labelled sizes
- Consider changing CIFAR-10 tranformations to be more similar to other references, normalization can match actual pixel value distribution. Try running and check if results improve

Maybe consider:
* Making smarter budget strategies, so that they have no overlap? What I mean is that two budget strategies cannot operate in the same budget space. Discuss pros and cons with group.
* Adding ability to select different models, and different datasets. Dataset names need to account for this, and both the selected model and dataset names need to be saved in the config file. Allows comparison of AL-algorithms on different models and datasets. Could be especially useful for determing what types of data the AL-algorithms work best on. Only implement this if HPC can handle many different models and datasets, otherwise if we only try a few different datasets create separate scripts for each.
"""


def reconstruct_rel_file_path_from_config(
    relative_save_folder,
    TRAIN_VAL_RATIO,
    EPOCHS,
    seeds,
    train_full_dataset_baseline,
    label_batch_sizes,
    RATIO_BATCH_TO_INIT_DATASET,
    selected_abbreviations,
    NUM_TRAIN_AL_ITERATIONS
):
    # Reconstruct the filename
    file_name = (
        f"model_{int(TRAIN_VAL_RATIO*100)}_{EPOCHS}_"
        f"seeds_{'-'.join(map(str, seeds))}_"
        f"baseline_{train_full_dataset_baseline}_"
        f"batch_{'-'.join(map(str, label_batch_sizes))}_"
        f"ratio_{RATIO_BATCH_TO_INIT_DATASET}_"
        f"algo_{'-'.join(selected_abbreviations)}_"
        f"iter_{NUM_TRAIN_AL_ITERATIONS}"
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
relative_save_folder = "./run_results"
TRAIN_VAL_RATIO = 0.8
EPOCHS = 3
seeds = [0, 1]
train_full_dataset_baseline = True

RATIO_BATCH_TO_INIT_DATASET = 3 # Ratio of initial labelled dataset size to label batch size (labels added per iteration)
BUDGET_STRATEGIES = {
   1: {"active": True, "batch_size": 200},
   2: {"active": True, "batch_size": 400}, 
   3: {"active": False, "batch_size": 800},
   4: {"active": False, "batch_size": 1600}
}
selected_strategies = [num for num, config in BUDGET_STRATEGIES.items() if config["active"]]
label_batch_sizes = [BUDGET_STRATEGIES[num]["batch_size"] for num in selected_strategies]

al_algorithms = ['random', 'uncertainty']
algorithm_abbreviations = {
    'random': 'ran',
    'uncertainty': 'unc',
    'margin': 'mar',      # Example additional algorithm, not implemented yet
    'entropy': 'ent'      # Example additional algorithm, not implemented yet
}
selected_abbreviations = [algorithm_abbreviations[algo] for algo in al_algorithms if algo in algorithm_abbreviations]
NUM_TRAIN_AL_ITERATIONS = 10
### END OF CONFIGURATION VARIABLES FOR FILE PATH RECONSTRUCTION ###


# Reconstruct the file path from the configuration variables
file_path = reconstruct_rel_file_path_from_config(
    relative_save_folder=relative_save_folder,
    TRAIN_VAL_RATIO=TRAIN_VAL_RATIO,
    EPOCHS=EPOCHS,
    seeds=seeds,
    train_full_dataset_baseline=train_full_dataset_baseline,
    label_batch_sizes=label_batch_sizes,
    RATIO_BATCH_TO_INIT_DATASET=RATIO_BATCH_TO_INIT_DATASET,
    selected_abbreviations=selected_abbreviations,
    NUM_TRAIN_AL_ITERATIONS=NUM_TRAIN_AL_ITERATIONS
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


print("\n\n\nAnalyzing simulation data...")
simulation_time = runtimes["simulation"]
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



