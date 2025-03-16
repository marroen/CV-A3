import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

# CIFAR-100 Loading with validation split
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_val_dataset_100 = torchvision.datasets.CIFAR100(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_dataset_100 = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Filters dataset to keep only the first 20 classes
def first_20_classes(dataset):

    first_20 = []
    for i, (image, label) in enumerate(dataset):
        if label < 20:
            first_20.append(i)

    return Subset(dataset, first_20)

train_val_dataset = first_20_classes(train_val_dataset_100)
test_dataset = first_20_classes(test_dataset_100)

# Split train into 80% train, 20% validation
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_subset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_subset, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

