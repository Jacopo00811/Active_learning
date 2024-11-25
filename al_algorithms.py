import torch
import numpy as np
from sklearn.cluster import kmeans_plusplus

# Uncertainty Sampling (Least Confidence)
def uncertainty_sampling_least_confidence(device, model, unlabelled_loader_relative, label_batch_size):
    model.eval()

    with torch.no_grad():
        # Pre-allocate a single tensor for all data
        all_uncertainties = []
        all_indices = []
        
        running_idx = 0
        for idx, (images, _) in enumerate(unlabelled_loader_relative):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            all_uncertainties.append(1 - torch.max(softmax_outputs, dim=1)[0])
            
            batch_size = images.size(0)
            indices = torch.arange(running_idx, running_idx + batch_size,device=device)
            all_indices.append(indices)

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


def uncertainty_sampling_margin(device, model, unlabelled_loader, label_batch_size):
    model.eval()
    uncertainties = []
    all_indices = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(unlabelled_loader):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            sorted_outputs, _ = torch.sort(softmax_outputs, dim=1, descending=True)
            margin = sorted_outputs[:, 0] - sorted_outputs[:, 1]
            uncertainties.extend(margin.cpu().numpy())
            all_indices.extend(list(range(idx * images.size(0), (idx * images.size(0)) + images.size(0))))

        uncertainties = torch.tensor(uncertainties)
        _, uncertain_indices = torch.topk(uncertainties, largest=False, k=label_batch_size)
        uncertain_indices = uncertain_indices.tolist()
        return [all_indices[i] for i in uncertain_indices]

def uncertainty_sampling_entropy(device, model, unlabelled_loader, label_batch_size):
    model.eval()
    uncertainties = []
    all_indices = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(unlabelled_loader):
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            log_softmax_outputs = torch.nn.functional.log_softmax(outputs, dim=1)
            entropy = -torch.sum(softmax_outputs * log_softmax_outputs, dim=1)
            uncertainties.extend(entropy.cpu().numpy())
            all_indices.extend(list(range(idx * images.size(0), (idx * images.size(0)) + images.size(0))))
    
        uncertainties = torch.tensor(uncertainties)
        _,uncertain_indices = torch.topk(torch.tensor(uncertainties), k=label_batch_size)
        uncertain_indices = uncertain_indices.tolist()
        return [all_indices[i] for i in uncertain_indices]

def hybrid_sampling_badge(device, model, unlabelled_loader, label_batch_size):
    def gradieant_embeddings(dataloader, model, device):
        """
        Extract features from the dataset using the backbone model.
        #TODO improve the function by doing batch processing
        """
        not_last_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        not_last_model.eval()
        embeddings = []
        indices = []

        with torch.no_grad():
            for idx, (images,_) in enumerate(dataloader):
                images = images.to(device)
                softmax_out = torch.nn.functional.softmax(model(images),dim=1)
                penultimate_features = not_last_model(images).squeeze()
                hypothetical_labels = torch.argmax(softmax_out, dim=1)
                one_hot = torch.zeros_like(softmax_out)
                one_hot.scatter_(1, hypothetical_labels.unsqueeze(1), 1)
                term_1 = softmax_out - one_hot  # [batch_size, num_classes]
                batch_embed = term_1.unsqueeze(-1)@penultimate_features.unsqueeze(1)
                batch_embed = batch_embed.reshape(images.size(0), -1)
                embeddings.extend(batch_embed.cpu())
                indices.extend(list(range(idx * images.size(0), (idx * images.size(0)) + images.size(0))))

        return embeddings, indices

    embeddings, image_indices = gradieant_embeddings(unlabelled_loader, model, device)
    embeddings = np.array(embeddings)
    _, indices = kmeans_plusplus(embeddings, n_clusters=label_batch_size)
    return [image_indices[i] for i in indices]

def Density_based_sampling_core_set(device, model, unlabelled_loader, label_batch_size, labeled_loader):
    labeled = []
    def get_embeddings(dataloader, model):
        model.eval()
        model_not_last = torch.nn.Sequential(*(list(model.children())[:-1]))
        model_not_last.eval()
        embeddings = []
        for _, (images,_) in enumerate(dataloader):
            images = images.to(device)
            penultimate_features = model_not_last(images).squeeze()
            embeddings.extend(penultimate_features)
        return embeddings
    
    labeled_embeddings = torch.vstack(get_embeddings(labeled_loader, model))
    unlabelled_embeddings = torch.vstack(get_embeddings(unlabelled_loader, model))

    while len(labeled) < label_batch_size:
        distances = torch.cdist(labeled_embeddings, unlabelled_embeddings,p=2)  #[label_size, unlabelled_size]
        min_distances = torch.min(distances, dim=0)
        _, max_min_distances = torch.topk(min_distances.values,k=label_batch_size)
        for i in max_min_distances:
            if i not in labeled:
                labeled.append(i)
                labeled_embeddings = torch.vstack([labeled_embeddings, unlabelled_embeddings[i]])
                break
    return labeled


## TODO: implement Uncertainty-Based Acquisition Functions : 
#           Margin Sampling DONE
#           Entropy Sampling DONE
#           Least Confidence Sampling DONE
#       Bayesian Uncertainty Sampling:
#           BALD Sampling             
#       Density-Based Methods
#           Core-Set  DONE
#      Hybrid Acquisition Functions
#           BADGE DONE
#      Batch Mode Active Learning
#           batch-margin
#           batch-BALD
#           cluster-margin