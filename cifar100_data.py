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

# Filters dataset to keep only the first 10 classes and the specified 10 labels
def filtered_classes(dataset):

    cifar_labels = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

    filtered_labels = []
    for i, (image, label) in enumerate(dataset):
        if label < 20 or label in cifar_labels:
            filtered_labels.append(i)

    return Subset(dataset, filtered_labels)

train_val_dataset = filtered_classes(train_val_dataset_100)
test_dataset = filtered_classes(test_dataset_100)

# Split train into 80% train, 20% validation
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# CHOICE TASK 5
# Takes a dataset, and adds a mirrored, upside down, and black and white version of each image to the dataset
def augment_dataset(dataset):

    augmented_images = []
    augmented_labels = []

    mirror = transforms.RandomHorizontalFlip(p=1.0)
    flip = transforms.RandomVerticalFlip(p=1.0)
    grayscale = transforms.Grayscale(num_output_channels=3)
    
    # Apply transformations on each image in dataset
    for image, label in dataset:

        mirrored = mirror(image)
        flipped = flip(image)
        grayscaled = grayscale(image)
        
        augmented_images.extend([image, mirrored, flipped, grayscaled])
        augmented_labels.extend([label, label, label, label])
    
    # Create a new dataset with augmented data
    augmented_dataset = torch.utils.data.TensorDataset(
        torch.stack(augmented_images),
        torch.tensor(augmented_labels)
    )
    
    return augmented_dataset

train_subset = augment_dataset(train_subset)

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

