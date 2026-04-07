from __future__ import annotations

from typing import Literal

from torch import nn
from torchvision import models

ModelName = Literal["resnet50", "densenet121"]


def _freeze_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _unfreeze_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def create_model(
    model_name: ModelName,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    num_outputs: int = 1,
) -> nn.Module:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_outputs)
        if freeze_backbone:
            _freeze_parameters(model)
            _unfreeze_parameters(model.fc)
        return model

    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_outputs)
        if freeze_backbone:
            _freeze_parameters(model)
            _unfreeze_parameters(model.classifier)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
