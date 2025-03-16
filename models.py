import torch
import torch.nn as nn

# LeNet-5 Model Definition A (avg. pooling)
class LeNet5A(nn.Module):
    def __init__(self):
        super(LeNet5A, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5) # 3 input channels (RGB), 6 output channels
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)           # 16 input channels, 5x5 feature map, 120 neurons
        self.fc2 = nn.Linear(120, 84)               # Dimensionality reduction 120 -> 84 neurons
        self.fc3 = nn.Linear(84, 10)                # 84 -> 10 class outputs
        self.relu = nn.ReLU()

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = x.view(x.size(0), -1)  # Flattens FC1 from 16x5x5 to 400
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# LeNet-5 Model Definition B (one more convolution layer)
class LeNet5B(nn.Module):
    def __init__(self):
        super(LeNet5B, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5) # 3 input channels (RGB), 6 output channels
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Additional convolution layer added down here
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)       # Flattens FC1 from 32x1x1 to 32
        self.fc2 = nn.Linear(120, 84)               # Dimensionality reduction 120 -> 84 neurons
        self.fc3 = nn.Linear(84, 10)                # 84 -> 10 class outputs
        self.relu = nn.ReLU()

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = self.relu(self.conv3(x)) # New conv
        x = x.view(x.size(0), -1)    # Flattens FC1 from 32x1x1 to 32
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# LeNet-5 Model Definition C (one more pooling layer)
class LeNet5C(nn.Module):
    def __init__(self):
        super(LeNet5C, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 3 input channels (RGB), 6 output channels
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Additional pooling layer added down here
        self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*5*5, 120)            # 16 input channels, 5x5 feature map, 120 neurons
        self.fc2 = nn.Linear(120, 84)                # Dimensionality reduction 120 -> 84 neurons
        self.fc3 = nn.Linear(84, 10)                 # 84 -> 10 class outputs
        self.relu = nn.ReLU()

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)      # New pooling
        x = x.view(x.size(0), -1)  # Flattens FC1 from 16x5x5 to 400
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# LeNet-5C model modified to produce 20 outputs
class LeNet5C_20(nn.Module):
    def __init__(self):
        super(LeNet5C_20, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
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
        x = self.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# LeNet-5C pretrained model with 10 outputs
# (identical to LeNetC, used to distinguish by class and for completeness)
class LeNet5C_pretrained(nn.Module):
    def __init__(self):
        super(LeNet5C_pretrained, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Additional pooling layer added down here
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
        x = self.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x