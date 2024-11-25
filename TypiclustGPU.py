# import torch
# from tqdm import tqdm
# from Model import MultiModel
# import torch.nn as nn
# from tqdm import tqdm  
# import matplotlib.pyplot as plt

# try:
#     import cupy as cp
#     import cuml
#     from cuml.cluster import KMeans as CuMLKMeans
#     CUML_AVAILABLE = True
# except ImportError:
#     CUML_AVAILABLE = False

# class Typiclust:
#     def __init__(self, backbone, initial_labeled_size, k_neighbors=20, device='cuda'):
#         """
#         Initialize Typiclust with the specified hyperparameters and number of clusters.
        
#         Args:
#             backbone: Model backbone to use
#             initial_labeled_size: Size of initial labeled set |L0|
#             k_neighbors: Number of neighbors for typicality computation
#             device: Device to run computations on
#         """
#         self.hyperparameters = {'number of classes': 10}
#         self.device = device
#         self.k_neighbors = k_neighbors
#         self.model = MultiModel(backbone=backbone, hyperparameters=self.hyperparameters, load_pretrained=True)
#         self.model.to(self.device)
#         self.feature_extractor = nn.Sequential(*list(self.model.pretrained_model.children())[:-1])
        
#         self.labeled_size = initial_labeled_size
#         self.discovered_clusters = set()
        
#     def cluster_features(self, features, budget):
#         """
#         Apply GPU-accelerated KMeans clustering to the extracted features.
        
#         Args:
#             features: Feature vectors (torch.Tensor on GPU)
#             budget: Number of samples to select (B)
#         """
#         n_clusters = self.labeled_size + budget
        
#         if CUML_AVAILABLE:
#             # Use cuML's GPU-accelerated KMeans
#             with tqdm(total=1, desc=f"GPU KMeans (|L|={self.labeled_size}, B={budget})", leave=False) as pbar:
#                 # Convert to cuML compatible format
#                 X = features.cpu().numpy()
#                 kmeans = CuMLKMeans(
#                     n_clusters=n_clusters,
#                     random_state=42,
#                     output_type='numpy'
#                 )
#                 cluster_labels = kmeans.fit_predict(X)
#                 pbar.update(1)
            
#             return torch.tensor(cluster_labels, device=self.device)
#         else:
#             # Fallback to PyTorch implementation if cuML is not available
#             with tqdm(total=1, desc=f"PyTorch KMeans (|L|={self.labeled_size}, B={budget})", leave=False) as pbar:
#                 cluster_labels = self._kmeans_pytorch(features, n_clusters)
#                 pbar.update(1)
            
#             return cluster_labels

#     def _kmeans_pytorch(self, features, n_clusters, max_iters=100, tol=1e-4):
#         """
#         Custom PyTorch implementation of KMeans that runs on GPU.
#         """
#         # Randomly initialize centroids
#         num_samples = features.shape[0]
#         indices = torch.randperm(num_samples)[:n_clusters]
#         centroids = features[indices].clone()
        
#         prev_centroids = torch.zeros_like(centroids)
#         for _ in range(max_iters):
#             # Calculate distances to centroids
#             distances = torch.cdist(features, centroids)
            
#             # Assign points to nearest centroid
#             cluster_labels = torch.argmin(distances, dim=1)
            
#             # Update centroids
#             prev_centroids.copy_(centroids)
#             for k in range(n_clusters):
#                 mask = (cluster_labels == k)
#                 if mask.any():
#                     centroids[k] = features[mask].mean(0)
            
#             # Check convergence
#             if torch.abs(prev_centroids - centroids).max() < tol:
#                 break
        
#         return cluster_labels

#     def active_learning_iteration(self, budget, unlabeled_dataloader):
#         """
#         Perform one iteration of active learning using TypiClust strategy with GPU acceleration.
#         """
#         # Extract features
#         features, _ = self.extract_features(unlabeled_dataloader)
        
#         # Keep features on GPU and normalize
#         features = features.to(self.device)
#         mean = features.mean(dim=0, keepdim=True)
#         std = features.std(dim=0, keepdim=True) + 1e-8
#         normalized_features = (features - mean) / std
        
#         # Perform clustering on GPU
#         cluster_labels = self.cluster_features(normalized_features, budget)
        
#         # Compute typicality (already GPU-optimized in your implementation)
#         typicality_scores = self.compute_typicality(normalized_features)
        
#         # Rest of the selection logic remains the same
#         cluster_sizes = [(label, (cluster_labels == label).sum().item()) 
#                         for label in range(cluster_labels.max().item() + 1)]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         selected_indices = []
#         for cluster, _ in cluster_sizes:
#             if cluster in self.discovered_clusters:
#                 continue
                
#             cluster_mask = (cluster_labels == cluster)
#             cluster_typicality = typicality_scores[cluster_mask]
            
#             top_sample_index = cluster_typicality.argmax()
#             original_index = torch.where(cluster_mask)[0][top_sample_index]
#             selected_indices.append(original_index.item())
            
#             self.discovered_clusters.add(cluster)
            
#             if len(selected_indices) >= budget:
#                 break
        
#         self.labeled_size += len(selected_indices)
#         return selected_indices




from Model import MultiModel
import torch
import torch.nn as nn
from torch_kmeans import KMeans
from tqdm import tqdm  
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
        
        self.labeled_size = initial_labeled_size
        self.discovered_clusters = set()

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
        k = min(self.k_neighbors + 1, len(features))
        _, indices = torch.topk(distances, k=k, largest=False)
        
        typicality = []
        for i in range(len(features)):
            neighbor_distances = distances[i, indices[i, 1:]]  # Exclude self
            avg_distance = torch.mean(neighbor_distances)
            typicality.append(1 / (avg_distance + 1e-8))
            
        return torch.tensor(typicality)

    def process_features_in_batches(self, features, kmeans, batch_size):
        """
        Process features in batches for KMeans clustering.
        """
        n_samples = features.size(0)
        all_labels = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = features[start_idx:end_idx]
            # Move batch to device, predict, then move back to CPU
            batch = batch.to(self.device)
            batch_labels = kmeans.predict(batch.unsqueeze(0)).squeeze(0)
            all_labels.append(batch_labels.cpu())
            
        return torch.cat(all_labels, dim=0)

    def cluster_features(self, features, budget, min_batch_size=256):
        """
        Apply KMeans clustering to the extracted features with memory-efficient batching.
        
        Args:
            features: Feature vectors
            budget: Number of samples to select (B)
            min_batch_size: Minimum size of batches to process
        """
        n_samples = features.size(0)
        
        # Calculate maximum possible number of clusters
        max_clusters = min(n_samples - 1, self.labeled_size + budget)
        n_clusters = max_clusters
        
        # Ensure batch_size is at least larger than n_clusters
        batch_size = max(min_batch_size, n_clusters + 100)  # Add buffer of 100
        
        # If batch_size is too large, we need to reduce n_clusters
        if batch_size > n_samples:
            batch_size = n_samples
            n_clusters = batch_size - 50  # Reduce clusters to ensure we have enough samples
        
        print(f"Using {n_clusters} clusters with batch size {batch_size}")
        
        # Initialize KMeans with the valid number of clusters
        kmeans = KMeans(
            n_clusters=n_clusters, 
            seed=42, 
            device=self.device
        )
        
        # Process the entire dataset in manageable chunks
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = features[start_idx:end_idx].to(self.device)
            
            # Only fit on first batch
            if start_idx == 0:
                kmeans.fit(batch.unsqueeze(0))
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Get cluster assignments for all samples in batches
        return self.process_features_in_batches(features, kmeans, batch_size).numpy()

    def active_learning_iteration(self, budget, unlabeled_dataloader):
        """
        Perform one iteration of active learning using TypiClust strategy.
        """
        # Extract and normalize features
        features, _ = self.extract_features(unlabeled_dataloader)
        
        # Move computations to CPU for better memory management
        features = features.cpu()
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-8
        normalized_features = (features - mean) / std
        
        # Perform clustering with appropriate batch size
        cluster_labels = torch.tensor(
            self.cluster_features(normalized_features, budget),
            device='cpu'  # Keep on CPU initially
        )
        
        # Compute typicality scores on CPU
        typicality_scores = self.compute_typicality(normalized_features)
        
        # Move necessary tensors to GPU only when needed
        cluster_labels = cluster_labels.to(self.device)
        typicality_scores = typicality_scores.to(self.device)
        
        # Select samples
        unique_clusters = torch.unique(cluster_labels)
        cluster_sizes = [(label.item(), (cluster_labels == label).sum().item()) 
                        for label in unique_clusters]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = []
        for cluster, _ in cluster_sizes:
            if cluster in self.discovered_clusters:
                continue
                
            cluster_mask = (cluster_labels == cluster)
            cluster_typicality = typicality_scores[cluster_mask]
            
            if len(cluster_typicality) > 0:
                top_sample_index = cluster_typicality.argmax()
                original_index = torch.where(cluster_mask)[0][top_sample_index]
                selected_indices.append(original_index.item())
                self.discovered_clusters.add(cluster)
            
            if len(selected_indices) >= budget:
                break
        
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
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as transforms

# Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

hyperparameters = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'num_workers': 2,
    'number of classes': 10,
    'backbone': 'resnet152',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'number of samples': 5,
}

print("Device:", hyperparameters['device'])
print("Device number:", torch.cuda.current_device())
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),     # subtract 0.5 and divide by 0.5
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)

unlabeled_dataloader = DataLoader(train_set, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=hyperparameters['num_workers'], drop_last=True)

# Initialize Typiclust
typiclust = Typiclust('resnet152', 600, hyperparameters['device'])
budget = 100

for i in range(4):
    selected_indices = typiclust.active_learning_iteration(budget, unlabeled_dataloader)
    print(f"Newly labeled samples: {selected_indices}, iteration {i+1}")
    # plot_top_n_typical_samples(selected_indices, train_set)
    budget += 10