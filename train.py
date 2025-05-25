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
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import shutil


def set_random_seeds(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


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

def align_loss(inputs, targets, EPS=1e-12):
    """Alignment loss for contrastive learning"""
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def calculate_loss(proto_features, pooled, out, ys, net_normalization_multiplier,
                   pretrain=False, criterion=nn.NLLLoss(), align_weight=5.0,
                   tanh_weight=2.0, class_weight=2.0, EPS=1e-10):
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

    return loss, acc, a_loss.item(), tanh_loss.item()

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


@torch.no_grad()
def evaluate(net, dataloader, device, epoch):
    """Evaluate model performance"""
    net.eval()

    total_correct = 0
    total_samples = 0
    cm = np.zeros((net._num_classes, net._num_classes), dtype=int)

    # Metrics
    total_sim_scores = 0.0
    total_nonzero_patches = 0.0
    local_size_total = 0.0
    abstained = 0

    progress_bar = tqdm(dataloader, desc=f"Eval Epoch {epoch}")

    for xs, ys in progress_bar:
        xs, ys = xs.to(device), ys.to(device)

        # Forward pass
        _, pooled, out = net(xs, inference=True)

        # Predictions
        max_scores, preds = torch.max(out, dim=1)
        abstained += (max_scores == 0).sum().item()

        # Update confusion matrix
        for pred, true in zip(preds.cpu().numpy(), ys.cpu().numpy()):
            cm[true][pred] += 1

        # Calculate metrics
        total_correct += (preds == ys).sum().item()
        total_samples += ys.size(0)

        # Prototype statistics
        repeated_weight = net._classification.weight.unsqueeze(1).repeat(1, pooled.shape[0], 1)
        sim_scores = torch.count_nonzero(torch.gt(torch.abs(pooled * repeated_weight), 1e-3).float(), dim=2).float()
        local_size = torch.count_nonzero(
            torch.gt(torch.relu((pooled * repeated_weight) - 1e-3).sum(dim=1), 0.).float(), dim=1
        ).float()

        total_sim_scores += sim_scores.sum().item()
        total_nonzero_patches += torch.count_nonzero(torch.gt(pooled, 1e-3), dim=1).float().sum().item()
        local_size_total += local_size.sum().item()

        # Update progress bar
        acc = total_correct / total_samples
        progress_bar.set_postfix({'acc': f'{acc:.3f}'})

    # Calculate final metrics
    accuracy = total_correct / total_samples
    avg_sim_scores = total_sim_scores / total_samples
    avg_nonzero_patches = total_nonzero_patches / total_samples
    avg_local_size = local_size_total / total_samples

    # Count active prototypes
    num_active_prototypes = torch.gt(net._classification.weight, 1e-3).any(dim=0).sum().item()
    sparsity = (torch.numel(net._classification.weight) -
                torch.count_nonzero(torch.relu(net._classification.weight - 1e-3)).item()) / \
               torch.numel(net._classification.weight)

    print(f"\nEpoch {epoch} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Abstained samples: {abstained}")
    print(f"Active prototypes: {num_active_prototypes}/{net._num_prototypes}")
    print(f"Sparsity ratio: {sparsity:.4f}")
    print(f"Avg similarity scores: {avg_sim_scores:.2f}")
    print(f"Avg nonzero patches: {avg_nonzero_patches:.2f}")
    print(f"Avg local size: {avg_local_size:.2f}")

    return accuracy, num_active_prototypes, sparsity

def get_patch_size(img_size=224, wshape=26):
    """Calculate patch size based on image and feature map sizes"""
    patchsize = 32
    skip = round((img_size - patchsize) / (wshape - 1))
    return patchsize, skip


def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    """Convert latent location to image patch coordinates"""
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        # Special handling for 26x26 feature maps
        h_coor_min = max(0, (h_idx - 1) * skip + 4)
        if h_idx < softmaxes_shape[-1] - 1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize

        w_coor_min = max(0, (w_idx - 1) * skip + 4)
        if w_idx < softmaxes_shape[-1] - 1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx * skip
        h_coor_max = min(img_size, h_idx * skip + patchsize)
        w_coor_min = w_idx * skip
        w_coor_max = min(img_size, w_idx * skip + patchsize)

    # Handle edge cases
    if h_idx == softmaxes_shape[1] - 1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] - 1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size - patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size - patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max


@torch.no_grad()
def visualize_topk_prototypes(net, projectloader, device, save_dir, k=10, img_size=224):
    """Visualize top-k most similar patches for each prototype"""
    net.eval()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    patchsize, skip = get_patch_size(img_size, 26)
    classification_weights = net._classification.weight

    # Collect top-k patches for each prototype
    topks = {}
    imgs = projectloader.dataset.imgs

    print("Collecting top-k patches for each prototype...")

    for i, (xs, ys) in enumerate(tqdm(projectloader, desc="Collecting patches")):
        xs = xs.to(device)

        # Get prototype activations
        pfs, pooled, _ = net(xs, inference=True)
        pooled = pooled.squeeze(0)
        pfs = pfs.squeeze(0)

        for p in range(pooled.shape[0]):
            # Only consider prototypes relevant to some class
            c_weight = torch.max(classification_weights[:, p])
            if c_weight > 1e-3:
                if p not in topks:
                    topks[p] = []

                # Update top-k list
                if len(topks[p]) < k:
                    topks[p].append((i, pooled[p].item()))
                else:
                    topks[p] = sorted(topks[p], key=lambda x: x[1], reverse=True)
                    if topks[p][-1][1] < pooled[p].item():
                        topks[p][-1] = (i, pooled[p].item())

    # Visualize top-k patches for each prototype
    print("Creating visualizations...")

    all_prototype_patches = []

    for p in tqdm(topks.keys(), desc="Visualizing prototypes"):
        prototype_patches = []

        for idx, score in sorted(topks[p], key=lambda x: x[1], reverse=True):
            if score > 0.1:  # Only include meaningful similarities
                # Load image
                img_path = imgs[idx][0] if isinstance(imgs[idx], tuple) else imgs[idx]
                image = Image.open(img_path).convert('RGB')
                image = transforms.Resize((img_size, img_size))(image)

                # Get activation location
                xs = projectloader.dataset[idx][0].unsqueeze(0).to(device)
                pfs, _, _ = net(xs, inference=True)
                pfs = pfs.squeeze(0)

                # Find max activation location
                max_val, max_idx = torch.max(pfs[p].flatten(), dim=0)
                h_idx = max_idx // pfs.shape[2]
                w_idx = max_idx % pfs.shape[2]

                # Extract patch
                h_min, h_max, w_min, w_max = get_img_coordinates(
                    img_size, pfs.shape, patchsize, skip, h_idx.item(), w_idx.item()
                )

                patch = image.crop((w_min, h_min, w_max, h_max))
                prototype_patches.append(patch)

        if prototype_patches:
            # Create grid for this prototype
            grid_size = int(np.ceil(np.sqrt(len(prototype_patches))))
            grid_img = Image.new('RGB', (grid_size * patchsize, grid_size * patchsize), 'white')

            for i, patch in enumerate(prototype_patches):
                row = i // grid_size
                col = i % grid_size
                grid_img.paste(patch, (col * patchsize, row * patchsize))

            # Save individual prototype grid
            grid_img.save(os.path.join(save_dir, f'prototype_{p}_topk.png'))
            all_prototype_patches.append((p, prototype_patches))

    # Create summary visualization
    if all_prototype_patches:
        fig, axes = plt.subplots(min(5, len(all_prototype_patches)),
                                min(10, k),
                                figsize=(20, 10))
        if len(axes.shape) == 1:
            axes = axes.reshape(-1, 1)

        for i, (p, patches) in enumerate(all_prototype_patches[:5]):
            for j, patch in enumerate(patches[:10]):
                ax = axes[i, j] if axes.ndim > 1 else axes[j]
                ax.imshow(patch)
                ax.axis('off')
                if j == 0:
                    ax.set_ylabel(f'P{p}', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prototypes_summary.png'), dpi=150)
        plt.close()

    print(f"Visualizations saved to {save_dir}")
    return topks


@torch.no_grad()
def visualize_prediction(net, image_path, device, save_dir, img_size=224, classes=None):
    """Visualize prediction for a single image"""
    net.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()

    # Transform for model
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    softmaxes, pooled, out = net(img_tensor, inference=True)
    pred_class = torch.argmax(out, dim=1).item()
    pred_score = out[0, pred_class].item()

    # Get patch size info
    patchsize, skip = get_patch_size(img_size, softmaxes.shape[2])

    # Find contributing prototypes
    classification_weights = net._classification.weight[pred_class]
    pooled = pooled.squeeze(0)
    softmaxes = softmaxes.squeeze(0)

    # Calculate prototype contributions
    contributions = pooled * classification_weights
    top_prototypes = torch.argsort(contributions, descending=True)[:5]

    # Visualize top contributing prototypes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Show original image
    image_resized = transforms.Resize((img_size, img_size))(original_image)
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title(f'Original Image\nPredicted: {classes[pred_class] if classes else pred_class}\nScore: {pred_score:.3f}')
    axes[0, 0].axis('off')

    # Show top contributing prototypes
    for i, p_idx in enumerate(top_prototypes):
        if i >= 5:
            break

        p = p_idx.item()
        row = (i + 1) // 3
        col = (i + 1) % 3

        # Find max activation location
        max_val, max_idx = torch.max(softmaxes[p].flatten(), dim=0)
        h_idx = max_idx // softmaxes.shape[2]
        w_idx = max_idx % softmaxes.shape[2]

        # Get patch coordinates
        h_min, h_max, w_min, w_max = get_img_coordinates(
            img_size, softmaxes.shape, patchsize, skip, h_idx.item(), w_idx.item()
        )

        # Draw rectangle on image
        img_with_rect = image_resized.copy()
        draw = ImageDraw.Draw(img_with_rect)
        draw.rectangle([(w_min, h_min), (w_max, h_max)], outline='yellow', width=3)

        axes[row, col].imshow(img_with_rect)
        axes[row, col].set_title(f'Prototype {p}\nSim: {pooled[p]:.3f}, Weight: {classification_weights[p]:.3f}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_visualization.png'), dpi=150)
    plt.close()

    print(f"Prediction visualization saved to {save_dir}")


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


data_path='./data/CUB_200_2011/'  # Update this path
epochs_pretrain=10
epochs=60
batch_size=64
batch_size_pretrain=128
lr_net=0.0005
lr_block=0.0005
lr_classifier=0.05
freeze_epochs=10
device='cuda' if torch.cuda.is_available() else 'cpu'
seed=1
visualize_every=10

def get_optimizer_nn(net, weight_decay=0.0) -> torch.optim.Optimizer:
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

if __name__ == "__main__":
    # Set seeds
    set_random_seeds(seed)

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    else:
        print(f"Using device: {device}")

    # Load data
    print("\nLoading CUB-200-2011 dataset...")
    trainloader, trainloader_pretrain, testloader, projectloader, classes = \
        get_cub_dataloaders(data_path, batch_size, batch_size_pretrain)

    # Create model
    print("\nCreating PIP-Net model...")
    net = create_model(num_classes=len(classes))
    net = net.to(device)

    # Get feature shape
    with torch.no_grad():
        dummy_batch = next(iter(trainloader))[0][:1].to(device)
        proto_features, _, _ = net(dummy_batch)
        print(f"Prototype features shape: {proto_features.shape}")

    # Create optimizers
    # Separate parameters for different learning rates
    # params_backbone = []
    # params_add_on = []
    # for name, param in net._net.named_parameters():
    #     if 'features.7.2' in name:  # Last layer
    #         params_add_on.append(param)
    #     else:
    #         params_backbone.append(param)

    # optimizer_net = torch.optim.AdamW([
    #     {'params': params_backbone, 'lr': lr_net},
    #     {'params': params_add_on, 'lr': lr_block},
    #     {'params': net._add_on.parameters(), 'lr': lr_block * 10}
    # ])

    # optimizer_classifier = torch.optim.AdamW(
    #     net._classification.parameters(), lr=lr_classifier
    # )

    # # Learning rate schedulers
    # scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_net, T_max=len(trainloader_pretrain) * epochs_pretrain
    # )
    # scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer_classifier, T_0=10, eta_min=0.001
    # )

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net)

    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net, T_max=len(trainloader_pretrain) * epochs_pretrain
    )

    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_classifier, T_0=10, eta_min=0.001
    )

    # Training history
    history = {
        'pretrain_loss': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'num_prototypes': [],
        'sparsity': []
    }

    # Phase 1: Pretrain prototypes
    print("\n" + "="*50)
    print("PHASE 1: Pretraining Prototypes")
    print("="*50)

    for epoch in range(1, epochs_pretrain + 1):
        # Freeze backbone initially
        for param in params_backbone:
            param.requires_grad = False

        loss, _ = train_epoch(net, trainloader_pretrain, optimizer_net,
                            optimizer_classifier, scheduler_net, None,
                            device, epoch, pretrain=True)
        history['pretrain_loss'].append(loss)

        print(f"Pretrain Epoch {epoch}/{epochs_pretrain} - Loss: {loss:.4f}")

    # Save pre-trained model
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_net_state_dict': optimizer_net.state_dict(),
        'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
    }, 'pre-trained-model.pth')

    # Visualize pretrained prototypes
    # if epochs_pretrain > 0:
    #     print("\nVisualizing pretrained prototypes...")
    #     visualize_topk_prototypes(net, projectloader, device,
    #                                 './visualizations/pretrained_prototypes', k=10)

    # Phase 2: Full training
    print("\n" + "="*50)
    print("PHASE 2: Full Training")
    print("="*50)

    # Reset schedulers
    del optimizer_net
    del optimizer_classifier
    del params_to_freeze
    del params_to_train
    del params_backbone
    del scheduler_classifier
    del scheduler_net

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net)

    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_classifier, T_0=10, eta_min=0.001
    )

    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_net, T_max=len(trainloader) * epochs, eta_min=lr_net/100
    )

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
    # Determine if we should finetune (only train classifier)
        finetune = epoch <= 3

        if finetune:
            for param in net._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
        else:
            # Unfreeze backbone after freeze_epochs
            if epoch > freeze_epochs:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                # Only train add-on layers and classifier
                for param in params_to_freeze:
                    param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                for param in net._add_on.parameters():
                    param.requires_grad = True
                for param in params_to_train:
                    param.requires_grad = True
                for param in params_backbone:
                    param.requires_grad = False

        # Train
        loss, acc = train_epoch(net, trainloader, optimizer_net,
                                optimizer_classifier, scheduler_net,
                                scheduler_classifier, device, epoch,
                                pretrain=False, finetune=finetune)

        # Evaluate
        test_acc, num_protos, sparsity = evaluate(net, testloader, device, epoch)

        # Update history
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        history['test_acc'].append(test_acc)
        history['num_prototypes'].append(num_protos)
        history['sparsity'].append(sparsity)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_net_state_dict': optimizer_net.state_dict(),
                'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')

        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"Train Loss: {loss:.4f}, Train Acc: {acc:.4f}")
        print(f"Test Acc: {test_acc:.4f} (Best: {best_acc:.4f})")
        print(f"Active Prototypes: {num_protos}, Sparsity: {sparsity:.4f}")

        # Periodic visualization
        if epoch % visualize_every == 0:
            print(f"\nVisualizing prototypes at epoch {epoch}...")
            visualize_topk_prototypes(net, projectloader, device,
                                    f'./visualizations/epoch_{epoch}_prototypes', k=10)

        # Print prototype weights per class
        if epoch % 10 == 0:
            print("\nPrototype importance per class:")
            for c in range(min(5, net._num_classes)):  # Show first 5 classes
                class_weights = net._classification.weight[c]
                important_protos = torch.where(class_weights > 0.1)[0]
                if len(important_protos) > 0:
                    print(f"Class {classes[c]}: {len(important_protos)} important prototypes")

        # Final visualization
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)