import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from models import LeNet5A, LeNet5B, LeNet5C
from cifar10_data import train_loader, val_loader, test_loader, test_dataset
from training_fns import train_model, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

models_to_train = {
    "LeNet5A": LeNet5A(),
    "LeNet5B": LeNet5B(),
    "LeNet5C": LeNet5C()
}

# Train and evaluate models
for model_name, model in models_to_train.items():
    print(f'\nTraining Model {model_name}')
    model = train_model(model, train_loader, val_loader, device, criterion, save=True)
    
    # First evaluate on validation set
    val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
    print(f'Model {model_name} Validation Results:')
    print(f'Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    # Then evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
    print(f'Model {model_name} Test Results:')
    print(f'Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

# Save the model weights after training
LeNet5C_weights = LeNet5C().to(device)
torch.save(LeNet5C_weights.state_dict(), "LeNet5C_weights.pth")

# Optional: Class-wise accuracy (using test set)
model = models_to_train["LeNet5C"]  # Example using model C
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print('\nClassification Report:')
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))