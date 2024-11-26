import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.transforms import v2 as transformsV2
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.tensorboard.writer import SummaryWriter
import itertools
import random


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

MEAN = np.array([0.5750, 0.6065, 0.6459])
STD = np.array([0.1854, 0.1748, 0.1794])
ROOT_DIRECTORY = "/zhome/f9/0/168881/Desktop/WindTurbineImagesCategorization/Data/Dataset"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")
hyper_parameters = {
    "network name": "MyNetwork",
    "input channels": 3,
    "number of classes": 5,
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "number of workers": 4,
    "epochs": 300,
    "epsilon": 1e-08,
    "weight decay": 1e-08,
    'beta1': 0.9,
    'beta2': 0.999,
    'learning rate': 0.001,
}

# logs_dir = 'WindTurbineImagesCategorization\\Network\\tests'
logs_dir = "runsS"
os.makedirs(logs_dir, exist_ok=True)
run_dir = os.path.join(logs_dir, f'run_')

transform = transformsV2.Compose([
    transformsV2.Pad(300, padding_mode="reflect"),
    transformsV2.RandomHorizontalFlip(p=0.5),
    transformsV2.RandomVerticalFlip(p=0.5),
    transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transformsV2.RandomAutocontrast(p=0.5),  
    transformsV2.RandomRotation(degrees=[0, 90]),
    transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
    transformsV2.CenterCrop(224),
    transformsV2.Resize((224, 224)), # Adjustable
    transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
    transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]) 

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
            labels -= 1  # Change the labels to start from 0
            labels = labels.type(torch.LongTensor)

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
                labels -= 1  # Change the labels to start from 0
                labels = labels.type(torch.LongTensor)

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
        {'Step_size': scheduler.step_size, 'Batch_size': hyper_parameters["batch size"], 'Gamma': hyper_parameters["gamma"]},
        {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
            f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
            f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
    )
    print('-------------- Finished training a new model! --------------\n')

    return {"Avg train loss": sum(all_train_losses)/len(all_train_losses), "Avg accuracy": sum(all_accuracies)/len(all_accuracies), "Avg val loss": sum(all_val_losses)/len(all_val_losses)}

# Define your hyperparameter grid
hyperparameter_grid = {
    'gamma': [0.8, 0.9],
    'batch size': [64, 128],
    'step size': [20, 25, 30],
}

def hyperparameter_search(loss_function, device, dataset_train, dataset_validation, hyperparameter_grid, missing_hp):
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
            dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for training with batch size: {hyper_parameters["batch size"]}")
        dataloader_validation = DataLoader(dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=True,
                                           num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for validation with batch size: {hyper_parameters["batch size"]}")
        model = MyNetwork(hyper_parameters)
        model.weight_initialization()
        model.to(DEVICE)

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

# Create Datasets and Dataloaders
dataset_train = MyDataset(root_directory=ROOT_DIRECTORY, mode="train",
                          transform=transform, split=hyper_parameters["split"], pca=False)
print(f"Created a new Dataset for training of length: {len(dataset_train)}")
dataset_validation = MyDataset(root_directory=ROOT_DIRECTORY,
                               mode="val", transform=None, split=hyper_parameters["split"], pca=False)
print(f"Created a new Dataset for validation of length: {len(dataset_validation)}\n")

loss_function = nn.CrossEntropyLoss()


# Perform hyperparameter search
all_combinations = create_combinations(hyperparameter_grid)
# random_samples = sample_hyperparameters(hyperparameter_grid, 10)

print(f"Number of combinations: {len(all_combinations)} (amount of models to test)\n\n")
best_hp = hyperparameter_search(
    loss_function, DEVICE, dataset_train, dataset_validation, all_combinations, hyper_parameters)

print(f"Best hyperparameters: {best_hp}")









"""
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as transforms
from torch.utils.tensorboard.writer import SummaryWriter
import itertools
import random
import torch.nn.functional as F
from Dataloader import *
from UNet import UNet
from torchsummary import summary
from Loss import *


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

def check_accuracy(model, dataloader, device, batch_size):
    model.eval()
    num_correct = 0
    num_pixels = 0

    image,label = [], []

    with torch.no_grad():
        for data in dataloader:

            image, label = data

            image = image.to(device)
            label = label.to(device)

            probs = F.sigmoid(model(image))

            predictions = (probs > 0.5).float()
            # Flatten tensors to compare pixel by pixel
            predictions = predictions.view(-1)
            label = label.view(-1)

            # Accumulate the number of correct pixels
            num_correct += (predictions == label).sum().item()

            # Accumulate the total number of pixels
            num_pixels += label.numel()

    # Calculate accuracy
    accuracy = num_correct / num_pixels
    
    print(
        f"Got {num_correct}/{num_pixels} with accuracy {accuracy * 100:.3f}%\n\n")
    model.train()
    return accuracy


def train_mod(model, logger, hyper_parameters, modeltype, device, loss_function, dataloader_train, dataloader_validation, dataloader_test, directory):

    optimizer, scheduler = set_optimizer_and_scheduler(hyper_parameters, model)


    epochs = hyper_parameters["epochs"]
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    validation_loss = 0

    images,labels = [], []

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
            loss_train = loss_function(labels, predicted_labels)
            loss_train.backward()
            optimizer.step()

            # Accumulate the loss and calculate the accuracy of predictions
            training_loss += loss_train.item()
            train_losses.append(loss_train.item())

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'Train loss', loss_train.item(
            ), epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'Train accuracy', train_accuracy, epoch*len(dataloader_train)+train_iteration)
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
                loss_val = loss_function(labels, output)

                validation_loss += loss_val.item()
                val_losses.append(loss_val.item())

                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            all_val_losses.append(sum(val_losses)/len(val_losses))

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        logger.add_scalars(f'Combined', {'Validation loss': validation_loss,
                                                 'Train loss': training_loss/len(dataloader_train)}, epoch)
        if scheduler is not None:
            scheduler.step()
            print(f"Current learning rate: {scheduler.get_last_lr()}")

    if scheduler is not None:
        logger.add_hparams(
            {f"Step_size": scheduler.step_size, f'Batch_size': hyper_parameters["batch size"], f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"], f'Loss function': hyper_parameters["loss"].__name__},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    else:
        logger.add_hparams(
            {f"Step_size": "None", f'Batch_size': hyper_parameters["batch size"], f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"], f'Loss function': hyper_parameters["loss"].__name__},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    
    
    # Check accuracy and save model
    accuracy = check_accuracy(model, dataloader_test, device, hyper_parameters['batch size'])
    save_dir = os.path.join(directory, f'accuracy_{accuracy:.3f}.pth')
    torch.save(model.state_dict(), save_dir)  # type: ignore

    return accuracy


def set_optimizer_and_scheduler(new_hp, model):
    if new_hp["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=new_hp["learning rate"],
                                     betas=(new_hp["beta1"],
                                            new_hp["beta2"]),
                                     weight_decay=new_hp["weight decay"],
                                     eps=new_hp["epsilon"])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=new_hp["learning rate"],
                                    momentum=new_hp["momentum"],
                                    weight_decay=new_hp["weight decay"])
    if new_hp["scheduler"] == "Yes":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
    else:
        scheduler = None
    return optimizer, scheduler


def hyperparameter_search(model, modeltype, device, dataset_train, dataset_validation, dataset_test, hyperparameter_grid, missing_hp, run_dir):
    best_performance = 0
    best_hyperparameters = None
    run_counter = 0
    modeltype_directory = os.path.join(run_dir, "PH2")
    modeltype_directory = os.path.join(modeltype_directory, f'{modeltype}')
    for hyper_parameters in hyperparameter_grid:
        # Empty memory before start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Current hyper parameters: {hyper_parameters}")
        hyper_parameters.update(missing_hp)
        # Initialize model, optimizer, scheduler, logger, dataloader
        dataloader_train = DataLoader(
            dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=True)
        print(f"Created a new Dataloader for training with batch size: {hyper_parameters['batch size']}")
        dataloader_validation = DataLoader(
            dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=False, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for validation with batch size: {hyper_parameters['batch size']}")
        dataloader_test = DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for testing with batch size: {hyper_parameters['batch size']}")

        log_dir = os.path.join(modeltype_directory, f'run_{str(run_counter)}_{hyper_parameters["network name"]}_{hyper_parameters["optimizer"]}_Scheduler_{hyper_parameters["scheduler"]}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)

        # Define the loss function
        loss_function = hyper_parameters["loss"]
        
        accuracy = train_mod(model, logger, hyper_parameters, modeltype, device,
                             loss_function, dataloader_train, dataloader_validation, dataloader_test, log_dir)

        run_counter += 1

        # Update best hyperparameters if the current model has better performance
        if accuracy > best_performance:
            best_performance = accuracy
            best_hyperparameters = hyper_parameters

        logger.close()
    print(f"\n\n############### Finished hyperparameter search! ###############")

    return best_hyperparameters


if torch.cuda.is_available():
    print("This code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToImage(),                          # Replace deprecated ToTensor()    
        transforms.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    ]
)

ph2_dataset = PH2(train=True, transform=transform)
ph2_train_size = int(0.8 * len(ph2_dataset))
ph2_val_size = len(ph2_dataset) - ph2_train_size
ph2_train, ph2_val = random_split(ph2_dataset, [ph2_train_size, ph2_val_size])
ph2_test = PH2(train=False, transform=transform) 

print(f"Created a new Dataset for training of length: {len(ph2_train)}")
print(f"Created a new Dataset for validation of length: {len(ph2_val)}")
print(f"Created a new Dataset for testing of length: {len(ph2_test)}")         


model = UNet(num_classes=1).to(device)
summary(model, (3, 256, 256))

print("Current working directory:", os.getcwd())

run_dir = "HPSearch"
os.makedirs(run_dir, exist_ok=True)


results = {}

hyperparameters = {
    'device': device, 
    'image size': (256, 256), 
    'backbone': 'UNet', 
    'torch home': 'TorchvisionModels', 
    'network name': 'HP_Search_PH2_UNet_1', 
    'dataset': 'PH2',
    'beta1': 0.9, 
    'beta2': 0.999, 
    'epsilon': 1e-08, 
    'number of workers': 2, 
    'weight decay': 0.0005,
    'epochs': 50,
    'optimizer': 'Adam',
    'scheduler': 'Yes',
    'momentum': 0.9, # Not used since SDG is not used
    }

hyperparameter_grid = {
    'batch size': [8, 16, 32],
    'step size': [5, 10, 20],
    'learning rate': [1e-2, 1e-3, 1e-4],
    'gamma': [0.8, 0.9],
    'loss': [bce_loss, dice_loss], 
    }


# ======================== Hyper parameter search =============================

samples = create_combinations(hyperparameter_grid)
# samples = sample_hyperparameters(hyperparameter_grid, 3)

print(f"Number of combinations: {len(samples)} (amount of models to test)\n\n")
best_hp = hyperparameter_search(model, hyperparameters["backbone"], device, ph2_train, ph2_val, ph2_test, samples, hyperparameters, run_dir)
results[hyperparameters["backbone"]] = best_hp
print(f"Best hyperparameters for {hyperparameters['backbone']}: {best_hp}")

print(f"\n\nResults: {results}")

"""