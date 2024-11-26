import torch
import numpy as np


# Uncertainty Sampling (Least Confidence)
def uncertainty_sampling_least_confidence(device, model, unlabelled_loader_relative, budget_query_size):
    model.eval()

    with torch.no_grad():
        # Pre-allocate a single tensor for all data
        all_uncertainties = []
        all_indices = []
        
        running_idx = 0
        for (images, _) in unlabelled_loader_relative:

            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)

            all_uncertainties.append(1 - torch.max(softmax_outputs, dim=1)[0])
            batch_size = images.size(0)
            indices = torch.arange(running_idx, running_idx + batch_size, device=device)
            all_indices.append(indices)
            running_idx += batch_size

        # Single concatenation at the end
        uncertainties = torch.cat(all_uncertainties)
        indices = torch.cat(all_indices)
        
        # Get top-k directly instead of full sort
        _, uncertain_indices = torch.topk(uncertainties, k=budget_query_size)
        return indices[uncertain_indices].tolist()

# Random Sampling Strategy
def random_sampling(unlabelled_loader_relative, budget_query_size):
    indices = list(range(len(unlabelled_loader_relative.dataset)))
    return np.random.choice(indices, budget_query_size, replace=False).tolist()

# Typiclus Sampling Strategy
def typiclust_sampling(model, typiclust_obj, unlabelled_loader_relative, budget):
    return typiclust_obj.active_learning_iteration(budget, unlabelled_loader_relative, model)