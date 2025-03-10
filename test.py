import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

# CIFAR-10 Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
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

batch_size = 32  # Use smaller batch_size if needed
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5C().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print epoch statistics
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Training finished')

# Testing Function
def test_model(model, test_loader, device):
    model.eval()  # Set to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

# Run testing after training
test_accuracy = test_model(model, test_loader, device)

# Optional: Class-wise accuracy
from sklearn.metrics import classification_report
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))