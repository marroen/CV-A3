import torch
import torchvision
import torchvision.transforms as transforms

# CIFAR-10 Loading with validation split
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_val_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

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

