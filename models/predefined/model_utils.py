# Define feature extractor
from typing import Iterable, Callable, Dict
from torch import Tensor
from torchvision import models
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from re import search


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return torchvision.transforms.functional.pad(image, padding, 0, "constant")


# Transferred ResNet Model
class PretrainedCustomResNet(nn.Module):
    def __init__(
        self,
        resnet: models.ResNet,
        num_classes: int,
        model_name: str,
        batch_norm: int = 512,
    ):
        super(PretrainedCustomResNet, self).__init__()
        # freezing parameters
        for param in resnet.parameters():
            param.requires_grad = False
        # convolutional layers of resnet34
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm1d(batch_norm)
        self.bn2 = nn.BatchNorm1d(batch_norm)
        self.fc1 = nn.Linear(batch_norm, batch_norm)
        self.fc2 = nn.Linear(batch_norm, num_classes)
        self.model_name = model_name

    def forward(self, x):
        embedded_features = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1, 1))(embedded_features)
        x = x.view(x.shape[0], -1)  # flattening
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        return x, embedded_features


# Transferred VGG Model
class PretrainedCustomVGG(nn.Module):
    def __init__(self, vgg: models.VGG, num_classes: int, model_name: str):
        super(PretrainedCustomVGG, self).__init__()
        # freezing parameters
        for param in vgg.features.parameters():
            param.requires_grad = False

        num_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1]
        vgg.classifier = nn.Sequential(*features)

        self.fc = nn.Linear(num_features, num_classes)
        self.model = vgg
        self.model_name = model_name

    def forward(self, x):
        embedded_features = self.model(x)
        x = self.fc(x)
        return x, embedded_features


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        layers: Iterable[str],
        model_name: str,
        classification_extraction=None,
    ):
        super().__init__()
        self.model = model
        self.layers = layers
        self.model_name = model_name
        self._classification_extraction = classification_extraction
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.model(x)

        # Certain pre-defined models return more than just the resulting probabilities
        if self._classification_extraction is not None:
            x = self._classification_extraction(x)

        return x, self._features


class PretrainedFeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        layers: Iterable[str],
        model_name: str,
        num_classes: int,
        classification_extraction=None,
        filter_condition=None,
    ):
        super().__init__()
        # freezing parameters
        if filter_condition is not None:
            for name, param in model.named_parameters():
                if filter_condition(name):
                    continue
                else:
                    param.requires_grad_ = False

        self.model = model
        self.model.aux_logits = False
        num_features = self.model.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)
        self.layers = layers
        self.model_name = model_name
        self._classification_extraction = classification_extraction
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.model(x)

        # Certain pre-defined models return more than just the resulting probabilities
        if self._classification_extraction is not None:
            x = self._classification_extraction(x)

        return x, self._features


class PretrainedCustomResNet(nn.Module):
    def __init__(
        self,
        resnet: models.ResNet,
        num_classes: int,
        model_name: str,
        filter_condition=lambda x: "layer4" in x or "fc" in x,
        batch_norm: int = 512,
    ):
        super(PretrainedCustomResNet, self).__init__()
        # freezing parameters
        for name, param in resnet.named_parameters():
            if filter_condition(name):
                continue
            else:
                param.requires_grad_ = False
        # convolutional layers of resnet34
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm1d(batch_norm)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(batch_norm, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.model_name = model_name

    def forward(self, x):
        embedded_features = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1, 1))(embedded_features)
        x = x.view(x.shape[0], -1)  # flattening
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        return x, embedded_features


def retrieve_models(num_classes):
    feature_models = [
        FeatureExtractor(
            model=models.resnet18(num_classes=num_classes),
            layers=["layer4"],
            model_name="ResNet18",
        ),
        FeatureExtractor(
            model=models.resnet34(num_classes=num_classes),
            layers=["layer4"],
            model_name="ResNet34",
        ),
        FeatureExtractor(
            model=models.resnet50(num_classes=num_classes),
            layers=["layer4"],
            model_name="ResNet50",
        ),
        FeatureExtractor(
            model=models.resnet101(num_classes=num_classes),
            layers=["layer4"],
            model_name="ResNet101",
        ),
        FeatureExtractor(
            model=models.resnet152(num_classes=num_classes),
            layers=["layer4"],
            model_name="ResNet152",
        ),
        # FeatureExtractor(
        #     model=models.alexnet(num_classes=num_classes),
        #     layers=["features"],
        #     model_name="AlexNet",
        # ),
        FeatureExtractor(
            model=models.googlenet(num_classes=num_classes),
            layers=["inception5b"],
            model_name="GoogleNet",
            classification_extraction=lambda x: x
            if isinstance(x, Tensor)
            else x.logits,
        ),
        FeatureExtractor(
            model=models.inception_v3(num_classes=num_classes, aux_logits=False),
            layers=["Mixed_7c"],
            model_name="InceptionV3",
            classification_extraction=lambda x: x
            if isinstance(x, Tensor)
            else x.logits,
        ),
        FeatureExtractor(
            model=models.vgg11(num_classes=num_classes),
            layers=["features"],
            model_name="VGG11",
        ),
        FeatureExtractor(
            model=models.vgg13(num_classes=num_classes),
            layers=["features"],
            model_name="VGG13",
        ),
        FeatureExtractor(
            model=models.vgg16(num_classes=num_classes),
            layers=["features"],
            model_name="VGG16",
        ),
        FeatureExtractor(
            model=models.vgg19(num_classes=num_classes),
            layers=["features"],
            model_name="VGG19",
        ),
    ]

    return feature_models
