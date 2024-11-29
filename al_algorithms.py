import torch
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



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
def random_sampling(unlabelled_loader_relative, budget_query_size, random_state):
    indices = list(range(len(unlabelled_loader_relative.dataset)))
    return random_state.choice(indices, size=budget_query_size, replace=False)

# Typiclus Sampling Strategy
def typiclust_sampling(model, typiclust_obj, unlabelled_loader_relative, budget):
    return typiclust_obj.active_learning_iteration(budget, unlabelled_loader_relative, model)


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

def hybrid_sampling_badge(device, model, unlabelled_loader, label_batch_size, random_state):
    def gradieant_embeddings(dataloader, model, device):
        """
        Extract features from the dataset using the backbone model.
        TODO make kmeansPlusPlus faster maybe by using pca to extract features??
        results with 200 components is 0.63852435
        results with 500 components is 0.8147309
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

        # pca = PCA(n_components=500, random_state=random_state)
        # normalizer = StandardScaler()
        # embeddings = np.vstack(embeddings)
        # embeddings = normalizer.fit_transform(embeddings)
        # embeddings = pca.fit_transform(embeddings)
        
        # print(embeddings.shape)
        # print(np.cumsum(pca.explained_variance_ratio_))
        return embeddings, indices

    embeddings, image_indices = gradieant_embeddings(unlabelled_loader, model, device)
    embeddings = np.array(embeddings)
    # print(embeddings.shape)
    # print(type(embeddings))
    _, indices = kmeans_plusplus(embeddings, n_clusters=label_batch_size, random_state=random_state)
    return [image_indices[i] for i in indices]

def Density_based_sampling_core_set(device, model, unlabelled_loader, label_batch_size, labeled_loader, random_state):
    labeled = []
    def get_embeddings(dataloader, model):
        model.eval()
        model_not_last = torch.nn.Sequential(*(list(model.children())[:-1]))
        model_not_last.eval()
        embeddings = []
        with torch.no_grad():
            for _, (images,_) in enumerate(dataloader):
                images = images.to(device)
                penultimate_features = model_not_last(images).squeeze()
                embeddings.append(penultimate_features.cpu().numpy())
        embeddings = np.vstack(embeddings)
        pca = PCA(n_components=50, random_state=random_state)        
        normalizer = StandardScaler()
        embeddings = normalizer.fit_transform(embeddings)
        embeddings = pca.fit_transform(embeddings)
        return embeddings
    
    labeled_embeddings = get_embeddings(labeled_loader, model)
    unlabelled_embeddings = get_embeddings(unlabelled_loader, model)
    labeled_embeddings = torch.from_numpy(labeled_embeddings).float()  # Convert to float32
    unlabelled_embeddings = torch.from_numpy(unlabelled_embeddings).float()


    while len(labeled) < label_batch_size:
        distances = torch.cdist(labeled_embeddings, unlabelled_embeddings,p=2)  #[label_size, unlabelled_size]        
        min_distances = torch.min(distances, dim=0)
        _, max_min_distances = torch.topk(min_distances.values,k=label_batch_size)
        for i in max_min_distances:
            if i not in labeled:
                labeled.append(i.item())
                labeled_embeddings = torch.vstack([labeled_embeddings, unlabelled_embeddings[i]])
                break
    return labeled

