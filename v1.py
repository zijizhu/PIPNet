import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import random
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import shutil

class TwoAugSupervisedDataset(Dataset):
    """Dataset that returns two augmented versions of each image"""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)

class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    """Custom augmentation without color changes"""
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    """Custom augmentation without shape changes"""
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


def get_cub_dataloaders(data_path='./data/CUB_200_2011/', batch_size=64,
                        batch_size_pretrain=128, img_size=224, num_workers=4):
    """Get data loaders for CUB dataset"""
    import tarfile
    import shutil

    # Normalization parameters
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    # Transform without augmentation
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Augmentation transforms
    transform1 = transforms.Compose([
        transforms.Resize(size=(img_size + 8, img_size + 8)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
    ])

    transform1_pretrain = transforms.Compose([
        transforms.Resize(size=(img_size + 32, img_size + 32)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
    ])

    transform2 = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Create datasets
    train_dir = os.path.join(data_path, 'dataset/train_crop')
    test_dir = os.path.join(data_path, 'dataset/test_crop')
    project_dir = os.path.join(data_path, 'dataset/train')

    # Training datasets
    trainset_base = torchvision.datasets.ImageFolder(train_dir)
    trainset = TwoAugSupervisedDataset(trainset_base, transform1, transform2)
    trainset_pretrain = TwoAugSupervisedDataset(trainset_base, transform1_pretrain, transform2)

    # Test and projection datasets
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    # Create dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True)
    trainloader_pretrain = DataLoader(trainset_pretrain, batch_size=batch_size_pretrain,
                                    shuffle=True, num_workers=num_workers, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)
    projectloader = DataLoader(projectset, batch_size=1, shuffle=False,
                             num_workers=num_workers)

    classes = trainset_base.classes
    print(f"Number of classes: {len(classes)}")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

    return trainloader, trainloader_pretrain, testloader, projectloader, classes


def convnext_tiny_26_features(pretrained=True):
    """ConvNeXt-Tiny with modified strides for 26x26 output"""
    from torchvision import models

    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)

    # Remove avgpool and classifier
    model.avgpool = nn.Identity()
    model.classifier = nn.Identity()

    # Modify strides to get 26x26 output
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.stride[0] == 2:
            if module.in_channels > 100:  # Skip early layers
                module.stride = (1, 1)

    return model

class NonNegLinear(nn.Module):
    """Linear layer with non-negative weights for interpretability"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.normal_(self.weight, mean=1.0, std=0.1)

    def forward(self, input):
        return F.linear(input, torch.relu(self.weight), self.bias)


class PIPNet(nn.Module):
    """Main PIP-Net model"""
    def __init__(self, num_classes, num_prototypes, feature_net,
                 add_on_layers, pool_layer, classification_layer):
        super().__init__()
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs, inference=False):
        # Extract features
        features = self._net(xs)

        # Apply prototype layer
        proto_features = self._add_on(features)

        # Pool to get prototype presence scores
        pooled = self._pool(proto_features)

        if inference:
            # During inference, ignore weak similarities
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)
            out = self._classification(clamped_pooled)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled)
            return proto_features, pooled, out

def create_model(num_classes=200, img_size=224):
    """Create PIP-Net model for CUB dataset"""
    # Feature extractor
    features = convnext_tiny_26_features(pretrained=True)

    # Get number of output channels
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, img_size, img_size)
        dummy_output = features(dummy_input)
        num_prototypes = dummy_output.shape[1]

    print(f"Number of prototypes: {num_prototypes}")

    # Additional layers
    add_on_layers = nn.Sequential(
        nn.Softmax(dim=1)  # Softmax over prototypes for each spatial location
    )

    # Pooling layer
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        nn.Flatten()
    )

    # Classification layer
    classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)

    # Create model
    model = PIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=features,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer
    )

    return model

def get_optimizer_nn(net, lr_net, lr_block, lr_classifier, weight_decay=0.0) -> torch.optim.Optimizer:
    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []

    print("chosen network is convnext", flush=True)
    for name,param in net._net.named_parameters():
        if 'features.7.2' in name:
            params_to_train.append(param)
        elif 'features.7' in name or 'features.6' in name:
            params_to_freeze.append(param)
        # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
        # elif 'features.5' in name or 'features.4' in name:
        #     params_backbone.append(param)
        # else:
        #     param.requires_grad = False
        else:
            params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)
    classification_weight = []
    classification_bias = []
    for name, param in net._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False

    paramlist_net = [
            {"params": params_backbone, "lr": lr_net, "weight_decay_rate": weight_decay},
            {"params": params_to_freeze, "lr": lr_block, "weight_decay_rate": weight_decay},
            {"params": params_to_train, "lr": lr_block, "weight_decay_rate": weight_decay},
            {"params": net._add_on.parameters(), "lr": lr_block*10., "weight_decay_rate": weight_decay}]

    paramlist_classifier = [
            {"params": classification_weight, "lr": lr_classifier, "weight_decay_rate": weight_decay},
            {"params": classification_bias, "lr": lr_classifier, "weight_decay_rate": 0},
    ]


    optimizer_net = torch.optim.AdamW(paramlist_net,weight_decay=weight_decay)
    optimizer_classifier = torch.optim.AdamW(paramlist_classifier,weight_decay=weight_decay)
    return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
