from typing import Tuple
import torch.nn as nn
from models.resnet import ResNet18, ResNet50


def get_model(model_name: str, num_classes: int = 1) -> nn.Module:
    """
    Get a neural network model by name with specified number of output classes

    Args:
        model_name: Model architecture ('resnet18', 'resnet50', or 'vgg19')
        num_classes: Number of output classes

    Returns:
        The initialized model
    """
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes=num_classes)
    elif model_name == 'vgg19':
        raise NotImplementedError("VGG19 not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: ['resnet18', 'resnet50', 'vgg19']")