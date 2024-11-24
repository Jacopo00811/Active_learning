import torch
import numpy as np


# Uncertainty Sampling (Least Confidence)
def uncertainty_sampling_least_confidence(device, model, unlabelled_loader_relative, label_batch_size):
    model.eval()

    with torch.no_grad():
        # Pre-allocate a single tensor for all data
        all_uncertainties = []
        all_indices = []
        
        for idx, (images, _) in enumerate(unlabelled_loader_relative):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            all_uncertainties.append(1 - torch.max(softmax_outputs, dim=1)[0])
            all_indices.append(torch.full((images.size(0),), idx, device=device))

        # Single concatenation at the end
        uncertainties = torch.cat(all_uncertainties)
        indices = torch.cat(all_indices)
        
        # Get top-k directly instead of full sort
        _, uncertain_indices = torch.topk(uncertainties, k=label_batch_size)
        return indices[uncertain_indices].tolist()

# Random Sampling Strategy
def random_sampling(unlabelled_loader_relative, label_batch_size):
    indices = list(range(len(unlabelled_loader_relative.dataset)))
    return np.random.choice(indices, label_batch_size, replace=False).tolist()

# Typiclus Sampling Strategy
def typiclus_sampling(model, unlabelled_loader_relative, budget):
    return model.active_learning_iteration(budget, unlabelled_loader_relative)