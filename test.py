import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms as transforms, torchvision
import matplotlib.pyplot as plt

def main():

        device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
        )
        print(f"Using {device} device")

        # Define transforms
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load dataset once (removed duplicate load)
        trainset = datasets.CIFAR10(root='./data', 
                                train=True,
                                download=True,
                                transform=transform)

        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_subset, val_subset = random_split(trainset, [train_size, val_size])

        # Create data loaders
        trainloader = DataLoader(train_subset, 
                        batch_size=32,  # Increased from 4 for better performance
                        shuffle=True,
                        num_workers=2)

        valloader = DataLoader(val_subset,
                        batch_size=32,
                        shuffle=False,  # No need to shuffle validation data
                        num_workers=2)

        # Visualization (now shows split sizes)
        images, labels = next(iter(trainloader))
        plt.figure(figsize=(10, 5))
        plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5)
        plt.title(f'Train samples ({len(train_subset)} images)\n' + 
                ' '.join(trainset.classes[label] for label in labels[:4]))
        plt.axis('off')
        plt.show()

        print(f"Training set size: {len(train_subset)} images")
        print(f"Validation set size: {len(val_subset)} images")

if __name__ == '__main__':
        main()