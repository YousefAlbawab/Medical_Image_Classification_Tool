import torch.nn as nn
import torch.nn.functional as F

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MedicalCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: [batch_size, 1, 256, 256]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 32, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 64, 64, 64]
        
        # Flatten the image
        x = x.view(-1, 64 * 64 * 64)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
