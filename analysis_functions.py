import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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


def plot_al_performance_across_budget_strategies(simulation_data, al_algorithm, confidence_level=0.95):
    """
    Plot average performance of AL algorithms across budget strategies with confidence intervals.

    Args:
        simulation_data (dict): Simulation data containing results and configuration
        al_algorithm (str): AL algorithm to plot
        confidence_level (float): Confidence level for confidence intervals
    
    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the plot
    """

    # Plot average performance of AL algorithms across budget strategies with confidence intervals.
    plt.figure(figsize=(10, 6))
    
    train_val_ratio = simulation_data["config"]["model"]["train_val_ratio"]
    
    for budget_strategy in simulation_data["config"]["budget_strategies"]:
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
        plt.plot(all_train_val_set_sizes[0], mean_test_accuracies, label=f"Budget Strategy {budget_strategy}", marker='o', linewidth=2)
        plt.fill_between(all_train_val_set_sizes[0], 
                        mean_test_accuracies - confidence_interval,
                        mean_test_accuracies + confidence_interval,
                        alpha=0.2)

    plt.xlabel("Training + Validation Set Size")
    plt.ylabel("Test Accuracy")
    plt.title(f"AL Performance Across Budget Strategies ({al_algorithm})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Return the figure
    return plt.gcf()