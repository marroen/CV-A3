import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import csv

# LeNet-5 Model Definition A
class LeNet5A(nn.Module):
    def __init__(self):
        super(LeNet5A, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)       # 3 input channels (RGB)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))       # [32,3,32,32] → [32,6,28,28]
        x = self.avg_pool1(x)             # → [32,6,14,14]
        x = self.relu(self.conv2(x))       # → [32,16,10,10]
        x = self.avg_pool2(x)             # → [32,16,5,5]
        x = x.view(x.size(0), -1)          # → [32,400]
        x = self.relu(self.fc1(x))         # → [32,120]
        x = self.relu(self.fc2(x))         # → [32,84]
        x = self.fc3(x)                    # → [32,10]
        return x
    
# LeNet-5 Model Definition B (one more convolution layer)
class LeNet5B(nn.Module):
    def __init__(self):
        super(LeNet5B, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)       # 3 input channels (RGB)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))       # [32,3,32,32] → [32,6,28,28]
        x = self.avg_pool1(x)             # → [32,6,14,14]
        x = self.relu(self.conv2(x))       # → [32,16,10,10]
        x = self.avg_pool2(x)             # → [32,16,5,5]
        x = self.relu(self.conv3(x))       # → [32,16,10,10]
        x = x.view(x.size(0), -1)          # → [32,400]
        x = self.relu(self.fc1(x))         # → [32,120]
        x = self.relu(self.fc2(x))         # → [32,84]
        x = self.fc3(x)                    # → [32,10]
        return x
    
# LeNet-5 Model Definition C (one more pooling layer)
class LeNet5C(nn.Module):
    def __init__(self):
        super(LeNet5C, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)       # 3 input channels (RGB)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Additional pooling layer added here
        self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))       # [32,3,32,32] → [32,6,28,28]
        x = self.avg_pool1(x)             # → [32,6,14,14]
        x = self.relu(self.conv2(x))       # → [32,16,10,10]
        x = self.avg_pool2(x)             # → [32,16,5,5]
        x = self.avg_pool3(x)             # New pooling → [32,16,5,5] (dimension preserved)
        x = x.view(x.size(0), -1)          # → [32,400]
        x = self.relu(self.fc1(x))         # → [32,120]
        x = self.relu(self.fc2(x))         # → [32,84]
        x = self.fc3(x)                    # → [32,10]
        return x

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

def train_model(index):
    # Model selection
    if index == 0:
        model = LeNet5A().to(device)
    elif index == 1:
        model = LeNet5B().to(device)
    elif index == 2:
        model = LeNet5C().to(device)
    else:
        raise ValueError("Invalid model index")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create CSV file and write header
    csv_filename = f"{model_name.lower()}_metrics.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])

    # Training Loop with validation
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        
        
        # Print epoch statistics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # Append metrics to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                f"{avg_train_loss:.4f}",
                f"{avg_val_loss:.4f}",
                f"{val_accuracy:.2f}"
            ])
        
        print(f"{model_name} Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")

    print('Training finished')
    return model

# Testing Function (works for both validation and test sets)
def evaluate_model(model, loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

models = []
# Train and evaluate models
for i, model_name in enumerate(['A', 'B', 'C']):
    print(f'\nTraining Model {model_name}')
    model = train_model(i)
    
    # First evaluate on validation set
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    print(f'Model {model_name} Validation Results:')
    print(f'Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    # Then evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f'Model {model_name} Test Results:')
    print(f'Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    models.append(model)

# Optional: Class-wise accuracy (using test set)
model = models[0]  # Example using model A
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