import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import random
from util_functions import reset_model_weights, train_model, evaluate_model
from torchvision import datasets, transforms
import torchvision.models as models
from enum import Enum
from tqdm import tqdm

class FineTuneMode(Enum):
    """ Indicates which layers we want to train during fine-tuning """
    NEW_LAYERS = 1
    CLASSIFIER = 2
    ALL_LAYERS = 3

class MultiModel(nn.Module):
    """ Custom class that wraps a torchvision model and provides methods to fine-tune """
    def __init__(self, hyperparameters, load_pretrained):
        super().__init__()
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []
        self.hyperparameters = hyperparameters

        if load_pretrained:
            self.pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.pretrained_model = models.resnet18(weights=None)
        
        self.classifier_layers = [self.pretrained_model.fc]
        self.pretrained_model.fc = nn.Linear(in_features=512, out_features=self.hyperparameters["number of classes"], bias=True)
        self.new_layers = [self.pretrained_model.fc]

    def forward(self, x):
        return self.pretrained_model(x)
    
    def fine_tune(self, mode: FineTuneMode):
        """ Fine-tune the model according to the specified mode using the requires_grad parameter """
        model = self.pretrained_model
        for parameter in model.parameters(): 
            parameter.requires_grad = False

        if mode is FineTuneMode.NEW_LAYERS:
            for layer in self.new_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        elif mode is FineTuneMode.CLASSIFIER:
            for layer in self.classifier_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        elif mode is FineTuneMode.ALL_LAYERS:
            for parameter in model.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(f"Invalid mode: {mode}")

        print(f"Ready to fine-tune the model, with the {mode} set to train")

    def count_parameters(self):
        total_params = sum(parameter.numel() for parameter in self.parameters())
        print(f"Total number of parameters: {total_params}")

class HPSearch:
    def __init__(self, train_val_dataset, test_loader, device, hyperparameters):
        self.train_val_dataset = train_val_dataset
        self.test_loader = test_loader
        self.device = device
        self.hyperparameters = hyperparameters
        
    def search(self, hp_grid, num_trials, epochs=10, train_val_ratio=0.8):
        """
        Run hyperparameter search using random search
        """
        best_hp = None
        best_accuracy = 0

        for trial in tqdm(range(num_trials), desc="Hyperparameter Search Trials"):
            print(f"\nTrial {trial+1}/{num_trials}")
            
            # Sample hyperparameters
            hp = {key: random.choice(values) for key, values in hp_grid.items()}
            
            # Update hyperparameters with sampled values
            self.hyperparameters.update(hp)
            
            # Split data
            train_size = int(len(self.train_val_dataset) * train_val_ratio)
            val_size = len(self.train_val_dataset) - train_size
            train_dataset, val_dataset = random_split(
                self.train_val_dataset, 
                [train_size, val_size]
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=hp['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=hp['batch_size'],
                shuffle=False
            )

            # Initialize model
            model = MultiModel(hyperparameters=self.hyperparameters, load_pretrained=True)
            model.to(self.device)
            reset_model_weights(model)

            # Train
            train_model(
                device=self.device,
                model=model,
                epochs=epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                hyperparameters=self.hyperparameters
            )

            # Evaluate
            accuracy = evaluate_model(
                device=self.device,
                model=model,
                data_loader=self.test_loader
            )

            print(f"HP: {hp}")
            print(f"Accuracy: {accuracy:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hp = hp

        return best_hp, best_accuracy

# Usage example:

# Load CIFAR-10 with transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 Dataset (Training and Test splits)
train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data loaders
full_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

hp_grid = {
    'batch_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'weight_decay': [0.0, 0.0001, 0.001],
    'optimizer': 'adam'
}


hyperparameters = {"number of classes": 10}

searcher = HPSearch(
    train_val_dataset=train_val_dataset,
    test_loader=full_test_loader,
    device=device,
    hyperparameters=hyperparameters
)

best_hp, best_acc = searcher.search(
    hp_grid=hp_grid,
    num_trials=len(hp_grid['batch_size']) * len(hp_grid['learning_rate']) * len(hp_grid['weight_decay']))

print(f"Best hyperparameters: {best_hp}")
print(f"Best accuracy: {best_acc:.2f}%")
