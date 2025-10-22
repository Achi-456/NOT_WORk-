import torch
import torch.nn as nn
import torch.nn.functional as F
from setup import NUM_CLASSES

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Squeeze
        y = self.global_avg_pool(x)  # [B, C, 1, 1]
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Scale
        return x * y

class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture with Batch Normalization, Dropout, and Channel Attention.
    Translated from Keras to PyTorch based on the provided architecture.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(ImprovedCNN, self).__init__()
        
        # Block 1: Conv2D(64) + BatchNorm + ReLU + ChannelAttention + MaxPool + Dropout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Block 2: Conv2D(128) + BatchNorm + ReLU + ChannelAttention + MaxPool + Dropout
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)
        
        # Block 3: Conv2D(256) + BatchNorm + ReLU + ChannelAttention + MaxPool + Dropout
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.ca3 = ChannelAttention(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Classifier Head
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Output Layer (9 classes for RealWaste dataset)
        self.fc_out = nn.Linear(512, num_classes)
        
        # Initialize weights with He normal (Kaiming normal)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.ca1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.ca2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.ca3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global Average Pooling and Classifier
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        x = self.fc_out(x)
        
        return x

def count_parameters(model):
    """Count the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN(num_classes=NUM_CLASSES).to(device)
    
    print(f"Model created successfully!")
    print(f"Number of parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
    
    # Test with random input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0, :5]}")  # First 5 logits of first sample
