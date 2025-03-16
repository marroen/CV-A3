from test import train_final_model, evaluate_model, LeNet5A, LeNet5B, LeNet5C
from choice3 import run_genetic_algorithm
from choice6 import extract_embeddings, plot_tsne, analyze_confusions
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

def main():
    # Main execution flow
    models = []
    best_hyperparams = []
    best_params_all = {
        'A': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'weight_decay': 0.0001,
            'batch_size': 32
        },
        'B': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'batch_size': 64
        },
        'C': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'weight_decay': 0.0,
            'batch_size': 32
        }
    }

    for _, model_class in enumerate([LeNet5A, LeNet5B, LeNet5C]):
        model_name = model_class.__name__[-1]  # Get 'A', 'B', or 'C'
        print(f"\n=== Optimizing {model_class.__name__} ===")
        
        # Genetic Algorithm Optimization
        best_params = run_genetic_algorithm(model_class, train_subset, val_subset, device, criterion)
        #best_params = best_params_all[model_name]
        best_hyperparams.append(best_params)
        print(f"Best parameters for {model_class.__name__}: {best_params}")
        
        # Final Training with Best Parameters
        model = train_final_model(train_val_dataset, model_class, best_params, device, criterion, save=True)
        models.append(model)
        
        # Test Evaluation
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=best_params['batch_size'], shuffle=False
        )
        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
        print(f"{model_class.__name__} Test Results: "
              f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        # Extract embeddings and plot
        embeddings, labels = extract_embeddings(model, test_loader, device)
        plot_tsne(embeddings, labels, test_dataset.classes, model_name)
        
        # Analyze potential confusions
        analyze_confusions(embeddings, labels, test_dataset.classes)

    # Generate classification report
    final_model = models[0]
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = final_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))


if __name__ == "__main__":
    main()