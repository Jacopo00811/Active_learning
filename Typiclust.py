from Model import MultiModel
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as transforms
from tqdm import tqdm  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class Typiclust:
    def __init__(self, hyperparameters, n_clusters=10, device='cuda'):
        """
        Initialize Typiclust with the specified hyperparameters and number of clusters.
        """
        self.hyperparameters = hyperparameters
        self.n_clusters = n_clusters
        self.device = device
        self.model = MultiModel(backbone=hyperparameters['backbone'], hyperparameters=hyperparameters, load_pretrained=True)
        self.model.to(self.device)
        self.feature_extractor = nn.Sequential(*list(self.model.pretrained_model.children())[:-1])  # Remove final FC layer

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

    def normalize_and_reduce(self, features):
        """
        Normalize the feature vectors.
        Args:
            features (torch.Tensor): Feature matrix.
        Returns:
            normalized_features (torch.Tensor): normalized feature matrix.
        """
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features.numpy())
        
        return torch.tensor(normalized_features, device=self.device)

    def cluster_features(self, features):
        """
        Apply KMeans clustering to the extracted features.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42, n_init=10)

        with tqdm(total=1, desc="KMeans Clustering", leave=False) as pbar:
            cluster_labels = kmeans.fit_predict(features.cpu().numpy())
            pbar.update(1)  # Update progress for clustering
        return cluster_labels, kmeans

    def fit(self, dataloader):
        """
        Perform feature extraction, normalization, PCA reduction, and clustering on the dataset.
        """
        print("Extracting features...")
        features, labels = self.extract_features(dataloader)
        print("Normalizing features...")
        normalized_features = self.normalize_and_reduce(features)
        print("Clustering features...")
        cluster_labels, kmeans = self.cluster_features(normalized_features)
        print("Clustering completed.")
        return normalized_features, labels, cluster_labels, kmeans, None

    def find_top_n_typical_samples(self, features, cluster_labels, kmeans, n=1):
        """
        Find the top N most typical samples for each cluster.
        """
        cluster_centers = torch.tensor(kmeans.cluster_centers_).to(self.device)
        features = features.cpu().numpy()  # Convert to NumPy for distance calculations

        typical_indices = {}
        for cluster_idx in range(kmeans.n_clusters):
            # Get indices of samples in this cluster
            cluster_samples = (cluster_labels == cluster_idx)
            cluster_features = features[cluster_samples]
            
            # Compute distances to the cluster center
            center = cluster_centers[cluster_idx].cpu().numpy()
            distances = torch.cdist(
                torch.tensor(cluster_features, device=self.device), 
                torch.tensor(center, device=self.device).unsqueeze(0)
            ).squeeze()
            
            # Get the indices of the N smallest distances
            closest_indices = distances.argsort()[:n]
            original_indices = np.where(cluster_samples)[0][closest_indices.cpu().numpy()]
            typical_indices[cluster_idx] = original_indices.tolist()
        
        return typical_indices

    def plot_top_n_typical_samples(self, typical_samples, dataset, n=1):
        """
        Plot the top N most typical samples for each cluster.
        Args:
            typical_samples (dict): Dictionary mapping cluster index to a list of N most typical sample indices.
            dataset (torchvision.datasets): The dataset containing the images.
            n (int): Number of most typical samples to display per cluster.
        """
        num_clusters = len(typical_samples)
        _, axes = plt.subplots(num_clusters, n, figsize=(n * 3, num_clusters * 3))

        for cluster_idx in range(num_clusters):
            for i, sample_idx in enumerate(typical_samples[cluster_idx]):
                image, _ = dataset[sample_idx]
                
                # Convert image back to 0-1 range if normalized
                image = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Reorder channels to HWC
                
                ax = axes[cluster_idx, i] if n > 1 else axes[cluster_idx]
                ax.imshow(image)
                ax.set_title(f"Cluster {cluster_idx}, Sample {i+1}")
                ax.axis("off")
        
        plt.tight_layout()
        plt.savefig("top_n_typical_samples.png")
        print("Saved the plot of top N typical samples for each cluster")


# Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

hyperparameters = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'num_workers': 2,
    'number of classes': 10,
    'input_size': 32 * 32 * 3,
    'hidden_size': 512,
    'output_size': 10,
    'backbone': 'resnet152',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'number of samples': 5,
}

print("Device:", hyperparameters['device'])
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),     # subtract 0.5 and divide by 0.5
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=hyperparameters['batch_size'],
                          shuffle=True, num_workers=hyperparameters['num_workers'], drop_last=True)

# Initialize Typiclust
typiclust = Typiclust(hyperparameters=hyperparameters, n_clusters=hyperparameters['number of classes'], device=hyperparameters['device'])

# Perform clustering
features, labels, cluster_labels, kmeans, pca = typiclust.fit(train_loader)

top_n_typical_samples = typiclust.find_top_n_typical_samples(features, cluster_labels, kmeans, n=hyperparameters['number of samples'])
# typical_samples = find_typical_samples(features, cluster_labels, kmeans)
print("Top N typical samples for each cluster:", top_n_typical_samples)

# Plot the most typical samples:
typiclust.plot_top_n_typical_samples(top_n_typical_samples, train_set, n=hyperparameters['number of samples'])
