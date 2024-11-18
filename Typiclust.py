from Model import MultiModel
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import v2 as transforms
from tqdm import tqdm  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Typiclust:
    def __init__(self, hyperparameters, labeled_dataloader, unlabeled_dataloader, n_clusters=10, k_neighbors=20, device='cuda'):
        """
        Initialize Typiclust with the specified hyperparameters and number of clusters.
        """
        self.hyperparameters = hyperparameters
        self.n_clusters = n_clusters
        self.device = device
        self.k_neighbors = k_neighbors
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.model = MultiModel(backbone=hyperparameters['backbone'], hyperparameters=hyperparameters, load_pretrained=True)
        self.model.to(self.device)
        self.feature_extractor = nn.Sequential(*list(self.model.pretrained_model.children())[:-1])  # Remove final FC layer
        
        # Initialize labeled and unlabeled sets
        self.labeled_set = []
        self.unlabeled_set = []


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
        
        Args:
            features (torch.Tensor): Feature matrix
        
        Returns:
            typicality (torch.Tensor): Typicality scores for each feature
        """
        # Compute pairwise distances
        distances = torch.cdist(features, features)
        
        # Find K-nearest neighbors for each point (excluding self)
        _, indices = torch.topk(distances, k=self.k_neighbors + 1, largest=False) # If largest is False then the k smallest elements are returned.
        
        # Compute average distance to K-nearest neighbors
        typicality = []
        for i in range(len(features)):
            neighbor_distances = distances[i, indices[i, 1:]]  # Exclude self
            avg_distance = torch.mean(neighbor_distances)
            typicality.append(1 / (avg_distance + 1e-8))  # Avoid division by zero
        
        return torch.tensor(typicality)

    def cluster_features(self, features):
        """
        Apply KMeans clustering to the extracted features.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42) #n_init=10

        with tqdm(total=1, desc="KMeans Clustering", leave=False) as pbar:
            cluster_labels = kmeans.fit_predict(features.cpu().numpy())
            pbar.update(1)  # Update progress for clustering
        return cluster_labels

    # def fit(self, dataloader):
    #     """
    #     Perform feature extraction, normalization, PCA reduction, and clustering on the dataset.
    #     """
    #     print("Extracting features...")
    #     features, labels = self.extract_features(dataloader)
    #     print("Normalizing features...")
    #     normalized_features = self.normalize_and_reduce(features)
    #     print("Clustering features...")
    #     cluster_labels, kmeans = self.cluster_features(normalized_features)
    #     print("Clustering completed.")
    #     return normalized_features, labels, cluster_labels, kmeans, None
    def active_learning_iteration(self, budget):
        """
        Perform one iteration of active learning using TypiClust strategy.
        """
        # Extract features from unlabeled data
        features, labels = self.extract_features(self.unlabeled_dataloader)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = torch.tensor(
            scaler.fit_transform(features.numpy()), 
            device=self.device
        )
        
        # Cluster the normalized features
        cluster_labels = torch.tensor(
            self.cluster_features(normalized_features), 
            device=self.device
        )

        # Compute typicality
        typicality_scores = self.compute_typicality(normalized_features).to(self.device)
        
        # Find largest uncovered clusters
        covered_clusters = set()
        if self.labeled_set:
            labeled_labels = labels[self.labeled_set]
            covered_clusters = set(labeled_labels)
        
        # Select samples from uncovered clusters
        selected_indices = []
        for cluster in range(self.n_clusters):
            if cluster in covered_clusters:
                continue
            
            # Find samples in this cluster
            cluster_mask = (cluster_labels == cluster)
            cluster_features = normalized_features[cluster_mask]
            cluster_typicality = typicality_scores[cluster_mask]
            
            # Select most typical sample from the cluster
            top_sample_index = cluster_typicality.argmax()
            original_index = torch.where(cluster_mask)[0][top_sample_index]
            selected_indices.append(original_index.item())
            
            # Stop if budget is reached
            if len(selected_indices) >= budget:
                break
        
        return selected_indices


    # def find_top_n_typical_samples(self, features, cluster_labels, kmeans, n=1):
    #     """
    #     Find the top N most typical samples for each cluster.
    #     """
    #     # cluster_centers = torch.tensor(kmeans.cluster_centers_).to(self.device)
    #     features = features.cpu().numpy()  # Convert to NumPy for distance calculations


    #     distances = torch.cdist(
    #             torch.tensor(cluster_features, device=self.device), 
    #             torch.tensor(center, device=self.device).unsqueeze(0)
    #         ).squeeze()

        # typical_indices = {}
        # for cluster_idx in range(kmeans.n_clusters):
        #     # Get indices of samples in this cluster
        #     cluster_samples = (cluster_labels == cluster_idx)
        #     cluster_features = features[cluster_samples]
            
        #     # Compute distances to the cluster center
        #     center = cluster_centers[cluster_idx].cpu().numpy()
        #     distances = torch.cdist(
        #         torch.tensor(cluster_features, device=self.device), 
        #         torch.tensor(center, device=self.device).unsqueeze(0)
        #     ).squeeze()
            
        #     # Get the indices of the N smallest distances
        #     closest_indices = distances.argsort()[:n]
        #     original_indices = np.where(cluster_samples)[0][closest_indices.cpu().numpy()]
        #     typical_indices[cluster_idx] = original_indices.tolist()
        
        # return typical_indices

    def plot_top_n_typical_samples(self, selected_indices, dataset, n=3):
        """
        Plot the top N most typical samples.
        Args:
            selected_indices (list): List of selected sample indices
            dataset (torchvision.datasets): The dataset containing the images
            n (int): Number of samples to display per cluster/row
        """
        # Determine number of rows needed
        num_rows = (len(selected_indices) + n - 1) // n
        
        # Create figure with appropriate number of rows and columns
        _, axes = plt.subplots(num_rows, n, figsize=(n * 3, num_rows * 3))
        
        # Flatten axes for easier indexing if multiple rows
        if num_rows > 1:
            axes = axes.flatten()
        
        # Plot selected samples
        for i, sample_idx in enumerate(selected_indices):
            image, label = dataset[sample_idx]
            
            # Convert image back to 0-1 range if normalized
            image = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Reorder channels to HWC
            
            # Select appropriate subplot
            ax = axes[i] if num_rows > 1 or n > 1 else axes
            ax.imshow(image)
            ax.set_title(f"Sample {i+1}, Label: {label}")
            ax.axis("off")
        
        # Hide any unused subplots
        for i in range(len(selected_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("top_typical_samples.png")
        print("Saved the plot of typical samples")

######################################## TESTING ########################################

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
# # Set the device to number 1
# # torch.cuda.set_device(1)
# print("Device number:", torch.cuda.current_device())
# transform = transforms.Compose([
#     transforms.ToImage(),
#     transforms.ToDtype(torch.float32, scale=True),     # subtract 0.5 and divide by 0.5
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train_set = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=False, transform=transform)
# labeled_dataloader = DataLoader(train_set, batch_size=hyperparameters['batch_size'],
#                           shuffle=True, num_workers=hyperparameters['num_workers'], drop_last=True)

# unlabeled_dataloader = DataLoader(train_set, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=hyperparameters['num_workers'], drop_last=True)


# # Initialize Typiclust
# typiclust = Typiclust(hyperparameters=hyperparameters, labeled_dataloader=labeled_dataloader, unlabeled_dataloader=unlabeled_dataloader, n_clusters=hyperparameters['number of classes'], device=hyperparameters['device'])

# for i in range(3):
#     selected_indices = typiclust.active_learning_iteration(10)

#     # Update labeled and unlabeled sets
#     typiclust.labeled_set.extend(selected_indices)
#     typiclust.unlabeled_set = [
#         idx for idx in typiclust.unlabeled_set 
#         if idx not in selected_indices
#     ]

#     print(f"Newly labeled samples: {selected_indices}, iteration {i+1}")

#     typiclust.plot_top_n_typical_samples(selected_indices, train_set, n=10)
