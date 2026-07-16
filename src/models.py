import torch.nn as nn
from torchvision.models import (
    VGG16_Weights,
    Wide_ResNet50_2_Weights,
    vgg16,
    wide_resnet50_2,
)

NUM_CLASSES = 10


def _build_head(in_features: int, num_classes: int) -> nn.Sequential:
    """New classifier head (paper Figure 2): FC(256) -> ReLU -> Dropout(0.5) -> FC(num_classes) -> LogSoftmax.

    Matches the head shape used in the authors' reference notebooks
    (https://github.com/raoofnaushad/EuroSAT_LULC), since the paper text
    doesn't specify exact FC sizes. LogSoftmax output means this must be
    paired with NLLLoss, not CrossEntropyLoss.
    """
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1),
    )


def _freeze_backbone(model: nn.Module, head: nn.Module) -> None:
    """Freeze every parameter except those belonging to the new head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True


def build_vgg16(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ImageNet-pretrained VGG16 with a frozen backbone and a new trainable head."""
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[6].in_features
    model.classifier[6] = _build_head(in_features, num_classes)
    _freeze_backbone(model, model.classifier[6])
    return model


def build_wide_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ImageNet-pretrained Wide ResNet-50-2 with a frozen backbone and a new trainable head."""
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = _build_head(in_features, num_classes)
    _freeze_backbone(model, model.fc)
    return model