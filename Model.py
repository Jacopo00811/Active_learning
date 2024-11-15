import torchvision.models as models
import torch.nn as nn
from enum import Enum


class FineTuneMode(Enum):
    """ Indicatea which layers we want to train during fine-tuning """
    " Just the new added layers " 
    NEW_LAYERS = 1
    " Just the classifier "
    CLASSIFIER = 2
    "Train all the layers "
    ALL_LAYERS = 3


class MultiModel(nn.Module):
    """ Custom class that wraps a torchvision model and provides methods to fine-tune """
    def __init__(self, backbone, hyperparameters, load_pretrained):
        super().__init__()
        self.backbone = backbone
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []
        self.hyperparameters = hyperparameters

        if backbone == "mobilenet_v3_large":
            if load_pretrained:
                self.pretrained_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            else:
                self.pretrained_model = models.mobilenet_v3_large(weights=None)
            
            self.classifier_layers = [self.pretrained_model.classifier]
            # Replace the final layer with a classifier for the exact number of classes
            self.pretrained_model.classifier[3] = nn.Linear(in_features=1280, out_features=self.hyperparameters["number of classes"], bias=True)
            self.new_layers = [self.pretrained_model.classifier[3]]
        elif backbone == "resnet152":
            if load_pretrained:
                self.pretrained_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            else:
                self.pretrained_model = models.resnet152(weights=None)
            
            self.classifier_layers = [self.pretrained_model.fc]
            self.pretrained_model.fc = nn.Linear(in_features=2048, out_features=self.hyperparameters["number of classes"], bias=True)
            self.new_layers = [self.pretrained_model.fc]
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

    def forward(self, x):
        return self.pretrained_model(x)
    
    def fine_tune(self, mode: FineTuneMode):
        " Fine-tune the model according to the specified mode using the requires_grad parameter "
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


# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(script_dir, 'TorchvisionModels')
# os.environ['TORCH_HOME'] = folder_path
# os.makedirs(folder_path, exist_ok=True)
# backbones = ["mobilenet_v3_large", "resnet152"]
# hyperparameters={"number of classes": 10    }
                 
# model = MultiModel(backbone=backbones[0], hyperparameters=hyperparameters, load_pretrained=False)
# model.count_parameters()