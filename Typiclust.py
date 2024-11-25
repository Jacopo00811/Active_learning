from Model import MultiModel
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Typiclust:
    def __init__(self, backbone, initial_labeled_size, device, k_neighbors=20):
        """
        Initialize Typiclust with the specified hyperparameters and number of clusters.
        
        Args:
            backbone: Model backbone to use
            initial_labeled_size: Size of initial labeled set |L0|
            k_neighbors: Number of neighbors for typicality computation
            device: Device to run computations on
        """
        self.hyperparameters = {'number of classes': 10}
        self.device = device
        self.k_neighbors = k_neighbors
        self.model = MultiModel(backbone=backbone, hyperparameters=self.hyperparameters, load_pretrained=True)
        self.model.to(self.device)
        self.feature_extractor = nn.Sequential(*list(self.model.pretrained_model.children())[:-1])
        
        # Track labeled set size and discovered clusters
        self.labeled_size = initial_labeled_size
        self.discovered_clusters = set()  # Keep track of clusters we've sampled from

    def extract_features(self, dataloader):
        """
        Extract features from the dataset using the backbone model.
        """
        self.model.eval()
        features, labels = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Feature Extraction", leave=False):
                inputs = inputs.to(self.device)
                outputs = self.feature_extractor(inputs).squeeze()
                features.append(outputs.cpu())
                labels.append(targets)

        return torch.cat(features, dim=0), torch.cat(labels, dim=0)
    
    def compute_typicality(self, features):
        """
        Compute typicality for each feature vector based on K-nearest neighbors.
        """
        distances = torch.cdist(features, features)
        _, indices = torch.topk(distances, k=self.k_neighbors + 1, largest=False)
        
        typicality = []
        for i in range(len(features)):
            neighbor_distances = distances[i, indices[i, 1:]]  # Exclude self
            avg_distance = torch.mean(neighbor_distances)
            typicality.append(1 / (avg_distance + 1e-8))  # Avoid division by zero
        
        return torch.tensor(typicality)

    def cluster_features(self, features, budget):
        """
        Apply KMeans clustering to the extracted features.
        
        Args:
            features: Feature vectors
            budget: Number of samples to select (B)
        """
        # Number of clusters = |Li-1| + B
        n_clusters = self.labeled_size + budget
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42) # Set random state for reproducibility
        
        with tqdm(total=1, desc=f"KMeans Clustering (|L|={self.labeled_size}, B={budget})", leave=False) as pbar:
            cluster_labels = kmeans.fit_predict(features.cpu().numpy())
            pbar.update(1)
        return cluster_labels

    def active_learning_iteration(self, budget, unlabeled_dataloader):
        """
        Perform one iteration of active learning using TypiClust strategy.
        
        Args:
            budget: Number of samples to select (B)
            
        Returns:
            list: Indices of selected samples
        """
        # Extract features from unlabeled data
        features, _ = self.extract_features(unlabeled_dataloader)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = torch.tensor(
            scaler.fit_transform(features.numpy()), 
            device=self.device
        )
        
        # Cluster the normalized features into |Li-1| + B clusters
        cluster_labels = torch.tensor(
            self.cluster_features(normalized_features, budget), 
            device=self.device
        )

        # Compute typicality
        typicality_scores = self.compute_typicality(normalized_features).to(self.device)
        
        # Get cluster sizes and sort by size
        cluster_sizes = [(label, (cluster_labels == label).sum().item()) 
                        for label in range(cluster_labels.max().item() + 1)]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = []
        for cluster, _ in cluster_sizes:
            # Skip if we've already sampled from this cluster
            if cluster in self.discovered_clusters:
                continue
                
            # Find samples in this cluster
            cluster_mask = (cluster_labels == cluster)
            cluster_typicality = typicality_scores[cluster_mask]
            
            # Select most typical sample from the cluster
            top_sample_index = cluster_typicality.argmax()
            original_index = torch.where(cluster_mask)[0][top_sample_index]
            selected_indices.append(original_index.item())
            
            # Mark cluster as discovered
            self.discovered_clusters.add(cluster)
            
            # Stop if budget is reached
            if len(selected_indices) >= budget:
                break
        
        # Update labeled set size for next iteration
        self.labeled_size += len(selected_indices)
        
        return selected_indices


def plot_top_n_typical_samples(selected_indices, dataset, images_per_row=10):
    """
    Plot the selected typical samples with exactly 10 images per row.
    
    Args:
        selected_indices (list): List of selected sample indices
        dataset (torchvision.datasets): The dataset containing the images
        images_per_row (int): Number of images to display per row (default=10)
    """
    # Calculate number of rows needed
    num_samples = len(selected_indices)
    num_rows = (num_samples + images_per_row - 1) // images_per_row
    
    # Create figure
    fig, axes = plt.subplots(num_rows, images_per_row, 
                            figsize=(images_per_row * 2, num_rows * 2))
    
    # Convert axes to 2D array if it's 1D (happens when num_rows=1)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot selected samples
    for idx, sample_idx in enumerate(selected_indices):
        # Calculate row and column position
        row = idx // images_per_row
        col = idx % images_per_row
        
        # Get and process image
        image, label = dataset[sample_idx]
        
        # Convert image back to 0-1 range if normalized
        image = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Reorder channels to HWC
        
        # Plot image
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"Sample {idx+1}\nLabel: {label}", fontsize=8)
        axes[row, col].axis("off")
    
    # Hide axes for unused slots
    for row in range(num_rows):
        for col in range(images_per_row):
            idx = row * images_per_row + col
            if idx >= num_samples:
                axes[row, col].axis('off')
                axes[row, col].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"top_{num_samples}_typical_samples.png", 
                bbox_inches='tight', 
                dpi=300)
    print(f"Saved plot of {num_samples} typical samples with {images_per_row} images per row")
    
    # Close the figure to free memory
    plt.close(fig)



######################################## TESTING ########################################
# from torch.utils.data import DataLoader, Subset
# import torchvision
# from torchvision.transforms import v2 as transforms

# # Empty memory before start
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# hyperparameters = {
#     'batch_size': 128,
#     'learning_rate': 0.001,
#     'num_epochs': 10,
#     'num_workers': 2,
#     'number of classes': 10,
#     'backbone': 'resnet152',
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     'number of samples': 5,
# }

# print("Device:", hyperparameters['device'])
# print("Device number:", torch.cuda.current_device())
# transform = transforms.Compose([
#     transforms.ToImage(),
#     transforms.ToDtype(torch.float32, scale=True),     # subtract 0.5 and divide by 0.5
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train_set = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=False, transform=transform)

# unlabeled_dataloader = DataLoader(train_set, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=hyperparameters['num_workers'], drop_last=True)

# # Initialize Typiclust
# typiclust = Typiclust(hyperparameters, unlabeled_dataloader, initial_labeled_size=100)
# budget = 10

# for i in range(4):
#     selected_indices = typiclust.active_learning_iteration(budget)
#     print(f"Newly labeled samples: {selected_indices}, iteration {i+1}")
#     plot_top_n_typical_samples(selected_indices, train_set)
#     budget += 10


# from Typiclust import Typiclust
# import torch

# class BayesianTypiclust(Typiclust):
#     def __init__(self, hyperparameters, labeled_dataloader, unlabeled_dataloader, n_clusters=10, 
#                  k_neighbors=20, device='cuda', acquisition_type='bald'):
#         super().__init__(hyperparameters, labeled_dataloader, unlabeled_dataloader, 
#                         n_clusters, k_neighbors, device)
#         self.acquisition_type = acquisition_type
        
#     def estimate_uncertainty(self, features, n_samples=10):
#         """Monte Carlo dropout for uncertainty estimation"""
#         self.model.train()  # Enable dropout
#         predictions = []
#         for _ in range(n_samples):
#             with torch.no_grad():
#                 pred = self.model(features)
#                 predictions.append(torch.softmax(pred, dim=1))
#         predictions = torch.stack(predictions)
#         mean = predictions.mean(0)
#         uncertainty = -(mean * torch.log(mean + 1e-10)).sum(1)  # Entropy
#         return uncertainty
    
#     def bald_acquisition(self, features, n_samples=10):
#         """Bayesian Active Learning by Disagreement"""
#         self.model.train()
#         outputs = torch.stack([torch.softmax(self.model(features), dim=1) 
#                              for _ in range(n_samples)])
#         mean = outputs.mean(0)
#         entropy1 = -(mean * torch.log(mean + 1e-10)).sum(1)
#         entropy2 = -(outputs * torch.log(outputs + 1e-10)).sum(2).mean(0)
#         return entropy1 - entropy2
    
#     def active_learning_iteration(self, budget):
#         features, labels = self.extract_features(self.unlabeled_dataloader)
#         normalized_features = self.normalize_and_reduce(features)
        
#         # Combine Bayesian uncertainty with typicality
#         uncertainty = self.bald_acquisition(normalized_features) if self.acquisition_type == 'bald' \
#                      else self.estimate_uncertainty(normalized_features)
#         typicality = self.compute_typicality(normalized_features)
        
#         # Weighted combination
#         scores = 0.5 * uncertainty + 0.5 * typicality
        
#         # Select samples with highest scores
#         selected_indices = torch.topk(scores, budget).indices
        
#         return selected_indices, normalized_features
    
# bayesian_typiclust = BayesianTypiclust(hyperparameters, 
#                                     labeled_dataloader,
#                                     unlabeled_dataloader,
#                                     acquisition_type='bald')
# selected_indices, features = bayesian_typiclust.active_learning_iteration(budget=10)