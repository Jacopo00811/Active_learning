from Model import MultiModel
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as transforms
from tqdm import tqdm  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score


class Typiclust:
    def __init__(self, hyperparameters, n_clusters=10, device='cuda'):
        """
        Initialize Typiclust with the specified hyperparameters and number of clusters.
        """
        self.hyperparameters = hyperparameters
        self.n_clusters = n_clusters
        self.device = device
        self.model = MultiModel(backbone=hyperparameters['backbone'], hyperparameters=hyperparameters, load_pretrained=False)
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

    def normalize_and_reduce(self, features, n_components=70):
        """
        Normalize the feature vectors and reduce dimensionality using PCA.
        Args:
            features (torch.Tensor): Feature matrix.
            n_components (int): Number of PCA components.
        Returns:
            reduced_features (torch.Tensor): Reduced feature matrix.
        """
        # Normalize the features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features.numpy())
        print("Initial component size:", normalized_features.shape)
        # Reduce dimensionality with PCA
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(normalized_features)
        # Calculate cumulative explained variance
        cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
        print("Final component size:", reduced_features.shape)
        return torch.tensor(reduced_features), pca, cumulative_explained_variance

    def cluster_features(self, features):
        """
        Apply KMeans clustering to the extracted features.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42, n_init=10)

        with tqdm(total=1, desc="KMeans Clustering", leave=False) as pbar:
            cluster_labels = kmeans.fit_predict(features.numpy())
            pbar.update(1)  # Update progress for clustering
        return cluster_labels, kmeans

    def fit(self, dataloader):
        """
        Perform feature extraction, normalization, PCA reduction, and clustering on the dataset.
        """
        print("Extracting features...")
        features, labels = self.extract_features(dataloader)
        print("Normalizing and reducing features...")
        # reduced_features, pca, explained_variance = self.normalize_and_reduce(features) ###
        # Print the cumulative explained variance
        # print("Cumulative explained variance ratio:", explained_variance) ####
        # plot_explained_variance(explained_variance) #
        print("Clustering features...")
        # cluster_labels, kmeans = self.cluster_features(reduced_features) ###
        cluster_labels, kmeans = self.cluster_features(features)

        print("Clustering completed.")
        # return reduced_features, labels, cluster_labels, kmeans, pca ###
        return features, labels, cluster_labels, kmeans, None


def find_top_n_typical_samples(features, cluster_labels, kmeans, n=1):
    """
    Find the top N most typical samples for each cluster.
    Args:
        features (torch.Tensor): Feature matrix (N x D), where N is the number of samples and D is the feature dimension.
        cluster_labels (numpy.ndarray): Cluster assignments for each sample.
        kmeans (KMeans): Trained KMeans model with cluster centers.
        n (int): Number of most typical samples to retrieve per cluster.
    Returns:
        typical_indices (dict): Dictionary mapping cluster index to a list of N most typical sample indices.
    """
    cluster_centers = torch.tensor(kmeans.cluster_centers_)  # Cluster centers from KMeans
    features = features.numpy()  # Convert to NumPy for distance calculations

    typical_indices = {}
    for cluster_idx in range(kmeans.n_clusters):
        # Get indices of samples in this cluster
        cluster_samples = (cluster_labels == cluster_idx)
        cluster_features = features[cluster_samples]
        
        # Compute distances to the cluster center
        center = cluster_centers[cluster_idx].numpy()
        distances = torch.cdist(torch.tensor(cluster_features), torch.tensor(center).unsqueeze(0)).squeeze()
        
        # Get the indices of the N smallest distances
        closest_indices = distances.argsort()[:n]
        original_indices = cluster_samples.nonzero()[0][closest_indices]
        typical_indices[cluster_idx] = original_indices.tolist()
    
    return typical_indices


def find_typical_samples(features, cluster_labels, kmeans):
    """
    Find the most typical samples for each cluster.
    Args:
        features (torch.Tensor): Feature matrix (N x D), where N is the number of samples and D is the feature dimension.
        cluster_labels (numpy.ndarray): Cluster assignments for each sample.
        kmeans (KMeans): Trained KMeans model with cluster centers.
    Returns:
        typical_indices (dict): Dictionary mapping cluster index to the most typical sample indices.
    """
    # Use L2-normalized features for consistency
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    cluster_centers = torch.nn.functional.normalize(torch.tensor(kmeans.cluster_centers_), p=2, dim=1)

    features = features.numpy()  # Convert to NumPy for distance calculations
    silhouette_avg = silhouette_score(features, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")

    typical_indices = {}
    for cluster_idx in range(kmeans.n_clusters):
        # Get indices of samples in this cluster
        cluster_samples = (cluster_labels == cluster_idx)
        cluster_features = features[cluster_samples]
        
        # Compute distances to the cluster center
        center = cluster_centers[cluster_idx].numpy()
        distances = torch.cdist(torch.tensor(cluster_features), torch.tensor(center).unsqueeze(0)).squeeze()
        
        # Find the sample with the smallest distance
        closest_idx = cluster_samples.nonzero()[0][distances.argmin()]
        typical_indices[cluster_idx] = closest_idx
    
    return typical_indices


def plot_pca_clusters(features, cluster_labels):
    """
    Plot PCA-reduced features with cluster labels.
    """
    plt.figure(figsize=(8, 6))
    for cluster_idx in range(max(cluster_labels) + 1):
        cluster_points = features[cluster_labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx}', alpha=0.6)
    
    plt.title("PCA-Reduced Features by Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig("pca_clusters.png")

def plot_explained_variance(explained_variance):
    """
    Plot the cumulative explained variance ratio.
    Args:
        explained_variance (numpy.ndarray): Cumulative explained variance ratio.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')    
    plt.title("Cumulative Explained Variance Ratio")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid()
    plt.savefig("explained_variance.png")
    




    
def plot_typical_samples(typical_samples, dataset):
    """
    Plot the most typical samples for each cluster.
    Args:
        typical_samples (dict): Dictionary mapping cluster index to the most typical sample index.
        dataset (torchvision.datasets): The dataset containing the images.
    """
    num_clusters = len(typical_samples)
    fig, axes = plt.subplots(1, num_clusters, figsize=(15, 5))
    
    for cluster_idx, ax in enumerate(axes):
        sample_idx = typical_samples[cluster_idx]
        image, label = dataset[sample_idx]
        
        # Convert image back to 0-1 range if normalized
        image = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Reorder channels to HWC
        
        ax.imshow(image)
        ax.set_title(f"Cluster {cluster_idx}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("typical_samples.png")

def plot_top_n_typical_samples(typical_samples, dataset, n=1):
    """
    Plot the top N most typical samples for each cluster.
    Args:
        typical_samples (dict): Dictionary mapping cluster index to a list of N most typical sample indices.
        dataset (torchvision.datasets): The dataset containing the images.
        n (int): Number of most typical samples to display per cluster.
    """
    num_clusters = len(typical_samples)
    fig, axes = plt.subplots(num_clusters, n, figsize=(n * 3, num_clusters * 3))

    for cluster_idx in range(num_clusters):
        for i, sample_idx in enumerate(typical_samples[cluster_idx]):
            image, label = dataset[sample_idx]
            
            # Convert image back to 0-1 range if normalized
            image = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Reorder channels to HWC
            
            ax = axes[cluster_idx, i] if n > 1 else axes[cluster_idx]
            ax.imshow(image)
            ax.set_title(f"Cluster {cluster_idx}, Sample {i+1}")
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("top_n_typical_samples.png")

def plot_pca_3d(features, cluster_labels, n_components=3):
    """
    Plot PCA-reduced features in 3D with cluster labels.
    Args:
        features (torch.Tensor): Reduced feature matrix (N x 3).
        cluster_labels (numpy.ndarray): Cluster assignments for each sample.
        n_components (int): Number of principal components used (default is 3).
    """
    # Prepare the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Update to avoid deprecated function warning
    colors = plt.colormaps['tab10']  # Use the recommended access method
    
    # Plot each cluster with different color
    for cluster_idx in range(max(cluster_labels) + 1):
        class_mask = cluster_labels == cluster_idx
        ax.scatter(features[class_mask, 0], features[class_mask, 1], features[class_mask, 2],
                   c=[colors(cluster_idx)], label=f'Cluster {cluster_idx}', alpha=0.6)
    
    ax.set_xlabel("PCA Component 1", fontweight="bold")
    ax.set_ylabel("PCA Component 2", fontweight="bold")
    ax.set_zlabel("PCA Component 3", fontweight="bold")
    ax.set_title("3D PCA-Reduced Features by Cluster", fontweight="bold", fontsize=20, color="red")
    
    # Optional: Rotate the plot for better visualization
    ax.view_init(30, 250)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig("pca_3d.png")




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

# Analyze clustering results
print("Cluster assignments:", cluster_labels[:10])

top_n_typical_samples = find_top_n_typical_samples(features, cluster_labels, kmeans, n=hyperparameters['number of samples'])
typical_samples = find_typical_samples(features, cluster_labels, kmeans)
print("Most typical samples for each cluster:", typical_samples)
print("Top N typical samples for each cluster:", top_n_typical_samples)

# Plot the most typical samples:
plot_typical_samples(typical_samples, train_set)
plot_top_n_typical_samples(top_n_typical_samples, train_set, n=hyperparameters['number of samples'])
plot_pca_clusters(features.numpy(), cluster_labels)
plot_pca_3d(features.numpy(), cluster_labels)   