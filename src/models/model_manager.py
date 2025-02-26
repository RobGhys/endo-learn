from typing import Tuple
import torch.nn as nn
from models.resnet import ResNet18, ResNet50
from models.vgg import VGG19
from models.vit import ViT_Base, ViT_Small
from models.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2


def get_model(model_name: str, num_classes: int = 1) -> nn.Module:
    """
    Get a neural network model by name with specified number of output classes

    Args:
        model_name: Model architecture ('resnet18', 'resnet50', 'vgg19', 'vit_base', 'vit_small',
                    'efficientnet_b0', 'efficientnet_b1', or 'efficientnet_b2')
        num_classes: Number of output classes

    Returns:
        The initialized model
    """
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes=num_classes)
    elif model_name == 'vgg19':
        return VGG19(num_classes=num_classes)
    elif model_name == 'vit_base':
        return ViT_Base(num_classes=num_classes)
    elif model_name == 'vit_small':
        return ViT_Small(num_classes=num_classes)
    elif model_name == 'efficientnet_b0':
        return EfficientNetB0(num_classes=num_classes)
    elif model_name == 'efficientnet_b1':
        return EfficientNetB1(num_classes=num_classes)
    elif model_name == 'efficientnet_b2':
        return EfficientNetB2(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: ['resnet18', 'resnet50', 'vgg19', 'vit_base', 'vit_small', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']")