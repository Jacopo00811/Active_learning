import json
import matplotlib.pyplot as plt

AL_ALGOS = ["random",
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
        }
        if algo_name == "baseline":
            plot_data["baseline"]["averaged_over_seed"] = algo_accuracies
            continue
        averaged_accuracies = [sum(x)/len(x) for x in zip(*algo_accuracies)]
        plot_data[algo_name]["averaged_over_seed"] = sum(averaged_accuracies)/len(averaged_accuracies)
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

plt.figure(figsize=(10, 5))
for algo_name, algo_data in plot_data.items():
    if algo_name == "baseline":
        plt.plot(algo_data["averaged_over_seed"], label=algo_name, linestyle='--')
    else:
        plt.plot(algo_data["averaged_over_seed"], label=algo_name)
plt.legend()


