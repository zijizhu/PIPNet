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
    for name,param in net.module._net.named_parameters():
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
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False

    paramlist_net = [
            {"params": params_backbone, "lr": lr_net, "weight_decay_rate": weight_decay},
            {"params": params_to_freeze, "lr": lr_block, "weight_decay_rate": weight_decay},
            {"params": params_to_train, "lr": lr_block, "weight_decay_rate": weight_decay},
            {"params": net.module._add_on.parameters(), "lr": lr_block*10., "weight_decay_rate": weight_decay}]

    paramlist_classifier = [
            {"params": classification_weight, "lr": lr_classifier, "weight_decay_rate": weight_decay},
            {"params": classification_bias, "lr": lr_classifier, "weight_decay_rate": 0},
    ]


    optimizer_net = torch.optim.AdamW(paramlist_net,weight_decay=weight_decay)
    optimizer_classifier = torch.optim.AdamW(paramlist_classifier,weight_decay=weight_decay)
    return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone

"""
Second debugging batch
"""
def align_loss(inputs, targets, EPS=1e-12):
    """Alignment loss for contrastive learning"""
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def calculate_loss(proto_features, pooled, out, ys,
                   align_weight, tanh_weight, class_weight, net_normalization_multiplier,
                   pretrain=False, finetune=False, criterion=nn.NLLLoss(), train_iter=None, print=True, EPS=1e-10):
    """Calculate combined loss for PIP-Net"""
    # Split augmented views
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    # Flatten spatial dimensions
    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    # Alignment loss
    a_loss = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.

    # Tanh loss for diversity
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean() +
                  torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()) / 2.

    """Adapted original loss"""
    # if not finetune:
    #     loss = align_weight*a_loss
    #     loss += tanh_weight * tanh_loss
    
    # if not pretrain:
    #     softmax_inputs = torch.log1p(out**net_normalization_multiplier)
    #     class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        
    #     if finetune:
    #         loss= class_weight * class_loss
    #     else:
    #         loss+= class_weight * class_loss
    
    # if pretrain:
    #     acc = 0.0
    # else:
    #     ys_combined = torch.cat([ys, ys])
    #     # Calculate accuracy
    #     ys_pred = torch.argmax(out, dim=1)
    #     acc = (ys_pred == ys_combined).float().mean().item()
    
    """Below is generated code"""
    if pretrain:
        # Only use alignment and tanh loss during pretraining
        loss = align_weight * a_loss + tanh_weight * tanh_loss
        acc = 0.0
    else:
        # Add classification loss
        ys_combined = torch.cat([ys, ys])
        softmax_inputs = torch.log1p(out ** net_normalization_multiplier)
        class_loss = criterion(F.log_softmax(softmax_inputs, dim=1), ys_combined)

        loss = align_weight * a_loss + tanh_weight * tanh_loss + class_weight * class_loss

        # Calculate accuracy
        ys_pred = torch.argmax(out, dim=1)
        acc = (ys_pred == ys_combined).float().mean().item()
    
    with torch.no_grad():
        if pretrain:
            train_iter.set_postfix_str(
            f'L: {loss.item():.3f}, LA:{a_loss.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
        else:
            if finetune:
                train_iter.set_postfix_str(
                f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
            else:
                train_iter.set_postfix_str(
                f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)

    return loss, acc

def train_epoch(net, dataloader, optimizer_net, optimizer_classifier,
                scheduler_net, scheduler_classifier, device, epoch,
                pretrain=False, finetune=False):
    """Train for one epoch"""
    net.train()

    # Configure gradients
    if pretrain:
        net._classification.requires_grad = False
    else:
        net._classification.requires_grad = True

    criterion = nn.NLLLoss(reduction='mean').to(device)

    total_loss = 0.0
    total_acc = 0.0

    # Training weights
    if pretrain:
        align_weight = (epoch / 10) * 1.0  # Gradually increase
        tanh_weight = 5.0
        class_weight = 0.0
    else:
        align_weight = 5.0
        tanh_weight = 2.0
        class_weight = 2.0

    print(f"{'Pretrain' if pretrain else 'Train'} Epoch {epoch} - "
          f"Align weight: {align_weight:.2f}, Tanh weight: {tanh_weight:.2f}, "
          f"Class weight: {class_weight:.2f}")

    progress_bar = tqdm(dataloader, desc=f"{'Pretrain' if pretrain else 'Train'} Epoch {epoch}")

    for i, (xs1, xs2, ys) in enumerate(progress_bar):
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

        # Zero gradients
        optimizer_net.zero_grad()
        optimizer_classifier.zero_grad()

        # Forward pass
        xs_combined = torch.cat([xs1, xs2])
        proto_features, pooled, out = net(xs_combined)

        # Calculate loss
        loss, acc, align_l, tanh_l = calculate_loss(
            proto_features, pooled, out, ys,
            net._classification.normalization_multiplier,
            pretrain=pretrain, criterion=criterion,
            align_weight=align_weight, tanh_weight=tanh_weight,
            class_weight=class_weight
        )

        # Backward pass
        loss.backward()

        # Update weights
        # if not finetune:
        #     optimizer_net.step()
        #     if scheduler_net:
        #         scheduler_net.step()

        # if not pretrain:
        #     optimizer_classifier.step()
        #     if scheduler_classifier:
        #         scheduler_classifier.step(epoch - 1 + (i / len(dataloader)))

        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i/len(dataloader)))

        if not finetune:
            optimizer_net.step()
            scheduler_net.step()

        if not pretrain:
            # Clamp small weights to zero
            with torch.no_grad():
                net._classification.weight.copy_(
                    torch.clamp(net._classification.weight.data - 1e-3, min=0.)
                )
                net._classification.normalization_multiplier.copy_(
                    torch.clamp(net._classification.normalization_multiplier.data, min=1.0)
                )
                if net._classification.bias is not None:
                    net.module._classification.bias.copy_(
                        torch.clamp(net._classification.bias.data, min=0.)
                    )

        # Update metrics
        total_loss += loss.item()
        total_acc += acc

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{acc:.3f}' if not pretrain else 'N/A',
            'align': f'{align_l:.3f}',
            'tanh': f'{tanh_l:.3f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader) if not pretrain else 0.0

    return avg_loss, avg_acc