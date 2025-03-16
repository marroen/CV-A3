import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from models import LeNet5C_pretrained
from cifar10_data import train_loader, val_loader, test_loader, test_dataset
from training_fns import train_model, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

models_to_train = {
    "LeNet5C_pretrained": LeNet5C_pretrained()
}

model_name = "LeNet5C_pretrained"

def create_pretrained():
    # Train and evaluate models
    for model_name, model in models_to_train.items():
        print(f'\nTraining Model {model_name}')

        model = LeNet5C_pretrained().to(device)
        pretrained_weights = torch.load("LeNet5C_20_weights.pth", map_location=device)

        # Remove last layer of pretrained weights
        pretrained_weights.pop('fc3.weight', None)
        pretrained_weights.pop('fc3.bias', None)

        model.load_state_dict(pretrained_weights, strict=False) # Load pretrained weights to new model

        return model
    
'''
model = create_pretrained()

# Save the model weights after training
LeNet5C_20_weights = LeNet5C_pretrained().to(device)
torch.save(LeNet5C_20_weights.state_dict(), "LeNet5C_pretrained_weights.pth")

# Optional: Class-wise accuracy (using test set)
model = models_to_train["LeNet5C_pretrained"]  # Example using model C
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
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))'''

