import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.transforms import v2 as transformsV2
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from torch.utils.tensorboard.writer import SummaryWriter
import itertools
import random
import torchvision


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

# Function to create all combinations of hyperparameters
def create_combinations(hyperparameter_grid):
    keys, values = zip(*hyperparameter_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# Function to randomly sample hyperparameters
def sample_hyperparameters(hyperparameter_grid, num_samples):
    samples = []
    for _ in range(num_samples):
        sample = {}
        for key, values in hyperparameter_grid.items():
            sample[key] = random.choice(values)
        samples.append(sample)
    return samples

def add_layer_weight_histograms(model, logger, model_name):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            logger.add_histogram(f"{model_name}/{name}.weights", module.weight)

def train_and_validate_net(model, loss_function, device, dataloader_train, dataloader_validation, optimizer, hyper_parameters, logger, scheduler, name="default"):
    epochs = hyper_parameters["epochs"]
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    validation_loss = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        """    Train step for one batch of data    """
        training_loop = create_tqdm_bar(
            dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')

        training_loss = 0
        model.train()  # Set the model to training mode
        train_losses = []
        accuracies = []
        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the parameter gradients for the current minibatch iteration

            images, labels = batch

            labels = labels.to(device)
            images = images.to(device)

            # Forward pass, backward pass and optimizer step
            predicted_labels = model(images)
            loss_train = loss_function(predicted_labels, labels)
            loss_train.backward()
            optimizer.step()

            # Accumulate the loss and calculate the accuracy of predictions
            training_loss += loss_train.item()
            train_losses.append(loss_train.item())

            # Uncomment to display images in tesnorboard
            # img_grid = make_grid(images)
            # logger.add_image(f"{name}/batch images", img_grid, epoch)

            # Add the model graph to tensorboard
            # if epoch == 0 and train_iteration == 0:
            #     logger.add_graph(model, images)

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)

            # features = images.reshape(images.shape[0], -1)
            # class_labels = [CLASSES[label] for label in predicted]  # predicted

            # if epoch > 47 and train_iteration == 5:  # Only the 5th iteration of each epoch after the 27th epoch
            #     logger.add_embedding(
            #         features, metadata=class_labels, label_img=images, global_step=epoch, tag=f'{name}/Embedding')

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'{name}/Train loss', loss_train.item(),
                              epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'{name}/Train accuracy', train_accuracy,
                              epoch*len(dataloader_train)+train_iteration)
        all_train_losses.append(sum(train_losses)/len(train_losses))
        all_accuracies.append(sum(accuracies)/len(accuracies))

        """    Validation step for one batch of data    """
        val_loop = create_tqdm_bar(
            dataloader_validation, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
        val_losses = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                images, labels = batch

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                output = model(images)

                # Calculate the loss
                loss_val = loss_function(output, labels)

                validation_loss += loss_val.item()
                val_losses.append(loss_val.item())

                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'{name}/Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            all_val_losses.append(sum(val_losses)/len(val_losses))

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        add_layer_weight_histograms(model, logger, name)
        logger.add_scalars(f'{name}/Combined', {'Validation loss': validation_loss,
                                                'Train loss': training_loss/len(dataloader_train)}, epoch)
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()}")
    # logger.add_hparams(
    #     {'Lr': scheduler.get_last_lr(
    #     )[0], 'Batch_size': hyper_parameters["batch size"], 'Gamma': hyper_parameters["gamma"]},
    #     {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
    #         f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
    #         f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
    # )
    logger.add_hparams(
        {'Step_size': scheduler.step_size, 'Batch_size': hyper_parameters["batch size"], 'Gamma': hyper_parameters["gamma"], 'Learning rate': hyper_parameters["learning rate"], 'Weight decay': hyper_parameters["weight decay"]},
        {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
            f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
            f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
    )
    print('-------------- Finished training a new model! --------------\n')

    return {"Avg train loss": sum(all_train_losses)/len(all_train_losses), "Avg accuracy": sum(all_accuracies)/len(all_accuracies), "Avg val loss": sum(all_val_losses)/len(all_val_losses)}

def hyperparameter_search(loss_function, device, dataset_train, dataset_validation, hyperparameter_grid, missing_hp, generator):
    # Initialize with a large value for minimization problems
    best_performance = float('inf')
    best_hyperparameters = None
    run_counter = 0
    for hyper_parameters in hyperparameter_grid:
        # Empty memory before start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Current hyper parameters: {hyper_parameters}")
        hyper_parameters.update(missing_hp)
        # Initialize model, optimizer, scheduler, logger, dataloader
        dataloader_train = DataLoader(
            dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False, generator=generator)
        print(f"Created a new Dataloader for training with batch size: {hyper_parameters["batch size"]}")
        dataloader_validation = DataLoader(dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=True,
                                           num_workers=hyper_parameters["number of workers"], drop_last=False, generator=generator)
        print(f"Created a new Dataloader for validation with batch size: {hyper_parameters["batch size"]}")
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(),
                               lr=hyper_parameters["learning rate"],
                               betas=(hyper_parameters["beta1"],
                                      hyper_parameters["beta2"]),
                               weight_decay=hyper_parameters["weight decay"],
                               eps=hyper_parameters["epsilon"])

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hyper_parameters["step size"], gamma=hyper_parameters["gamma"])

        logger = SummaryWriter(run_dir+str(run_counter)+f" batch size_{hyper_parameters['batch size']} lr_{
                               hyper_parameters['learning rate']} gamma_{hyper_parameters['gamma']}")

        # Train the model
        dict = train_and_validate_net(model, loss_function, device, dataloader_train, dataloader_validation,
                                      optimizer, hyper_parameters, logger, scheduler, name=hyper_parameters["network name"])

        run_counter += 1

        # Update best hyperparameters if the current model has better performance
        if dict["Avg val loss"] < best_performance:
            best_performance = dict["Avg val loss"]
            best_hyperparameters = hyper_parameters

        logger.close()
    print(f"\n\n############### Finished hyperparameter search! ###############")

    return best_hyperparameters

def set_global_seed(seed):
    """
    Set global seed for reproducibility in PyTorch, NumPy, and Python random.
    Ensures that random number generation is consistent across runs, reducing uncertainty in results.

    Args:
        seed (int): Seed to set for random number generation
    
    Returns:
        torch.Generator: Generator for use in data loaders
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

##################### Main code #####################
generator = set_global_seed(42)

MEAN = np.array([0.5, 0.5, 0.5])
STD = np.array([0.5, 0.5, 0.5])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device in use: {DEVICE}")

hyper_parameters = {
    "network name": "ResNet18Pretrained",
    "input channels": 3,
    "number of classes": 10,
    "number of workers": 4,
    "epochs": 25,
    "epsilon": 1e-08,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight decay': 1e-8, # TODO: Test this in another run
}

# Define your hyperparameter grid
hyperparameter_grid = {
    'gamma': [0.8, 0.9, 0.7],
    'batch size': [64, 128, 32],
    'step size': [5, 10],
    'learning rate': [0.001, 0.01, 0.0001],
}

logs_dir = "HpSearch"
os.makedirs(logs_dir, exist_ok=True)
run_dir = os.path.join(logs_dir, f'run_')

transform = transformsV2.Compose([
    transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
    transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]) 

TRAIN_VAL_RATIO = 0.8 

# Create Datasets and Dataloaders
train_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_size = int(TRAIN_VAL_RATIO * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

# Split the dataset into training and validation
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)
print(f"Created a new Dataset for training of length: {len(train_dataset)}")
print(f"Created a new Dataset for validation of length: {len(val_dataset)}\n")

loss_function = nn.CrossEntropyLoss()

# Perform hyperparameter search
all_combinations = create_combinations(hyperparameter_grid)
# random_samples = sample_hyperparameters(hyperparameter_grid, 10)

print(f"Number of combinations: {len(all_combinations)} (amount of models to test)\n\n")
best_hp = hyperparameter_search(
    loss_function, DEVICE, train_dataset, val_dataset, all_combinations, hyper_parameters, generator)
print(f"Best hyperparameters: {best_hp}")