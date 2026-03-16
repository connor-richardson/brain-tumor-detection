from torchvision import models
import torch.nn as nn

def create_model():
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    for param in model.features.parameters():
        param.requires_grad = False

    for layer in model.features[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        1
    )

    return model