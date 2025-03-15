import torch
import torch.nn as nn

# LeNet-5 Model Definition A (avg. pooling)
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
        # Additional convolution layer added here
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

# LeNet-5 Model Definition C (one more pooling layer)
class LeNet5C_20(nn.Module):
    def __init__(self):
        super(LeNet5C_20, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)       # 3 input channels (RGB)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Additional pooling layer added here
        self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20) # Modified output from 10 to 20
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
        x = self.fc3(x)                    # → [32,20]
        return x