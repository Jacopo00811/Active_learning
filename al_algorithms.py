import torch
import numpy as np


# Uncertainty Sampling (Least Confidence)
def uncertainty_sampling_least_confidence(device, model, unlabelled_loader_relative, label_batch_size):
    model.eval()
    uncertainties = []
    samples_indices = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(unlabelled_loader_relative):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            max_confidences, _ = torch.max(softmax_outputs, dim=1)
            uncertainties.extend(1 - max_confidences.cpu().numpy())
            samples_indices.extend([idx] * images.size(0))

    uncertain_indices = np.argsort(uncertainties)[-label_batch_size:]
    selected_indices = [samples_indices[i] for i in uncertain_indices]
    return selected_indices

# Random Sampling Strategy
def random_sampling(unlabelled_loader_relative, label_batch_size):
    indices = list(range(len(unlabelled_loader_relative.dataset)))
    return np.random.choice(indices, label_batch_size, replace=False).tolist()

# Typiclus Sampling Strategy
def typiclus_sampling(model, unlabelled_loader_relative, budget):
    return model.active_learning_iteration(budget, unlabelled_loader_relative)