import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

heatmap_data = ["random",
                "uncertainty",
                "typiclust",
                "margin",
                "entropy",
                "badge",
                "coreset"] 
FILES = ["cifar-10_resnet-18_ptw0_tvr80_ep10_bs32_sds0-1-2-3-4_bl1_algo-ran-unc-typ-mar-ent-bad-cor_bsr1_bis500_bqs250_bni20",
         "cifar-10_resnet-18_ptw0_tvr80_ep10_bs32_sds5-6-7-8-9-10-11-12-13-14_bl1_algo-ran-unc-typ-mar-ent-bad-cor_bsr1_bis500_bqs250_bni20",
         "cifar-10_resnet-18_ptw0_tvr80_ep10_bs32_sds15-16-17-18-19-20-21-22-23-24_bl1_algo-ran-unc-typ-mar-ent-bad-cor_bsr1_bis500_bqs250_bni20",
         "cifar-10_resnet-18_ptw0_tvr80_ep10_bs32_sds25-26-27-28-29-30-31-32-33-34_bl1_algo-ran-unc-typ-mar-ent-bad-cor_bsr1_bis500_bqs250_bni20",
         "cifar-10_resnet-18_ptw0_tvr80_ep10_bs32_sds35-36-37-38-39-40-41-42-43-44_bl1_algo-ran-unc-typ-mar-ent-bad-cor_bsr1_bis500_bqs250_bni20",]

def extract_accuracies(data,accuracies_dict):
    results = data.get("results", {})
    for seed, seed_data in results.items():
        for strategy, strategy_data in seed_data.items():
            if isinstance(strategy_data, dict):
                for algo_name, algo_data in strategy_data.items():
                    if algo_name not in accuracies_dict:
                        accuracies_dict[algo_name] = []
                    accuracies_dict[algo_name].append(algo_data.get("test_accuracies", []))
        if "baseline" not in accuracies_dict:
            accuracies_dict["baseline"] = []
        accuracies_dict["baseline"].append(seed_data.get("full_dataset_baseline", {}))
    return accuracies_dict

def prepare_plot_data(data_dict):
    plot_data = {}
    for algo_name, algo_accuracies in data_dict.items():
        plot_data[algo_name] = {
            "raw": algo_accuracies,
            "averaged_over_seed": [],
            "stddev_over_seed": [],
        }
        if algo_name == "baseline":
            plot_data["baseline"]["averaged_over_seed"] = algo_accuracies
            continue

        acc_array = np.array(algo_accuracies)
        
        # Calculate mean and std across seeds
        averaged_accuracies = np.mean(acc_array, axis=0)
        std_accuracies = np.std(acc_array, axis=0)
        
        plot_data[algo_name]["averaged_over_seed"] = averaged_accuracies.tolist()
        plot_data[algo_name]["stddev_over_seed"] = std_accuracies.tolist()


    return plot_data
print("-----------------")
dict_algos = {}
for file in FILES:
    with open("run_results/" + file , "r") as f:
        data = json.load(f)
    dict_algos = extract_accuracies(data,dict_algos)
print(dict_algos.keys())
print([len(v) for v in dict_algos.values()])
print(dict_algos["baseline"])
print("-----------------")

plot_data = prepare_plot_data(dict_algos)
print(plot_data.keys())
# Ploting the results

plt.figure(figsize=(12, 6))

# Plot each algorithm
for algo_name, algo_data in plot_data.items():
    if algo_name == "baseline":
        continue
    
    # Get data
    means = algo_data["averaged_over_seed"]
    stds = algo_data["stddev_over_seed"]
    x = range(len(means))
    
    # Plot mean line with points
    line = plt.plot(x, means, 
             label=algo_name,
             marker='o',
             markersize=6,
             linewidth=2,
             markerfacecolor='white',
             markeredgewidth=1.5)
    
    # Add confidence interval
    color = line[0].get_color()
    plt.fill_between(x, 
                    [m - 2*s for m, s in zip(means, stds)],
                    [m + 2*s for m, s in zip(means, stds)],
                    alpha=0.2,
                    color=color)

# Customize plot
plt.title('Performance Comparison of Active Learning Methods', fontsize=14, pad=15)
plt.xlabel('Active Learning Iterations', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# Set x-ticks every 2 iterations
plt.xticks(range(0, len(means), 2), fontsize=10)
plt.yticks(fontsize=10)

plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
plt.tight_layout()

plt.show()

heatmap_data = []
algos = []
for algo_name, algo_data in plot_data.items():
    if algo_name == "baseline":
        continue
    
    # Get data
    means = algo_data["averaged_over_seed"]
    heatmap_data.append(means)
    algos.append(algo_name)

heatmap = np.zeros((len(heatmap_data), len(heatmap_data)))

for i, algo1 in enumerate(heatmap_data):
    for j, algo2 in enumerate(heatmap_data):
        if i != j:
            t_stat, p_value = ttest_ind(algo1, algo2)
            heatmap[i, j] = p_value
        else:
            heatmap[i, j] = 1  # Diagonal should be 1 (no difference)
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
print(algos)
print(heatmap)