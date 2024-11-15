from Model import MultiModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as transforms
from tqdm import tqdm  

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

    def cluster_features(self, features):
        """
        Apply KMeans clustering to the extracted features.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        with tqdm(total=1, desc="KMeans Clustering", leave=False) as pbar:
            cluster_labels = kmeans.fit_predict(features.numpy())
            pbar.update(1)  # Update progress for clustering
        return cluster_labels, kmeans

    def fit(self, dataloader):
        """
        Perform feature extraction and clustering on the dataset.
        """
        print("Extracting features...")
        features, labels = self.extract_features(dataloader)
        print("Clustering features...")
        cluster_labels, kmeans = self.cluster_features(features)
        print("Clustering completed.")
        return features, labels, cluster_labels, kmeans





hyperparameters = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'num_workers': 0,
    'number of classes': 10,
    'input_size': 32 * 32 * 3,
    'hidden_size': 512,
    'output_size': 10,
    'backbone': 'resnet152',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

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
features, labels, cluster_labels, kmeans = typiclust.fit(train_loader)

# Analyze clustering results
print("Cluster assignments:", cluster_labels[:10])
