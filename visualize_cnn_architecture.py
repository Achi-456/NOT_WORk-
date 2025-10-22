"""
CNN Architecture Visualization Tool
====================================
This script creates comprehensive visualizations of the Deeper Residual CNN (ResNet-34 style) architecture.

Usage:
    python visualize_cnn_architecture.py

Outputs:
    - deeper_resnet34_architecture_diagram.png: Full network architecture
    - deeper_resnet34_block_details.png: Residual block and Channel Attention details
    - deeper_resnet34_parameter_distribution.png: Parameter distribution charts
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Model Definition
# ============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel-wise attention."""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with optional channel attention."""
    
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelAttention(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class DeeperResidualCNN(nn.Module):
    """
    Deeper Residual CNN based on ResNet-34 architecture.
    
    Architecture:
    - Stem: Conv(7x7) -> BN -> ReLU -> MaxPool
    - Layer 1: 3 residual blocks (64 channels)
    - Layer 2: 4 residual blocks (128 channels)
    - Layer 3: 6 residual blocks (256 channels)
    - Layer 4: 3 residual blocks (512 channels)
    - Global Average Pooling
    - Fully Connected Classifier
    
    Total: 16 residual blocks (ResNet-34 style: [3, 4, 6, 3])
    """
    
    def __init__(self, num_classes=9):
        super(DeeperResidualCNN, self).__init__()
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet-34 style: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        
        # Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_attention=True))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_attention=True))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ============================================================================
# Visualization Functions
# ============================================================================

def print_architecture_summary(model, device='cpu'):
    """Print detailed text-based architecture summary."""
    print("=" * 100)
    print("🏗️  DEEPER RESIDUAL CNN (ResNet-34 Style) - ARCHITECTURE VISUALIZATION")
    print("=" * 100)
    
    print("\n📐 LAYER-BY-LAYER BREAKDOWN:\n")
    print(f"{'Layer Name':<30} {'Type':<20} {'Output Shape':<20} {'Params':<15}")
    print("-" * 100)
    
    # Track shapes through the network
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Stem
    x = model.stem(dummy_input)
    stem_params = sum(p.numel() for p in model.stem.parameters())
    print(f"{'Stem (Conv7x7+BN+ReLU+Pool)':<30} {'Sequential':<20} {str(tuple(x.shape)):<20} {stem_params:,}")
    
    # Layer 1 (3 blocks)
    x = model.layer1(x)
    layer1_params = sum(p.numel() for p in model.layer1.parameters())
    print(f"{'Layer1 (3 ResBlocks)':<30} {'Sequential':<20} {str(tuple(x.shape)):<20} {layer1_params:,}")
    
    # Layer 2 (4 blocks)
    x = model.layer2(x)
    layer2_params = sum(p.numel() for p in model.layer2.parameters())
    print(f"{'Layer2 (4 ResBlocks)':<30} {'Sequential':<20} {str(tuple(x.shape)):<20} {layer2_params:,}")
    
    # Layer 3 (6 blocks)
    x = model.layer3(x)
    layer3_params = sum(p.numel() for p in model.layer3.parameters())
    print(f"{'Layer3 (6 ResBlocks)':<30} {'Sequential':<20} {str(tuple(x.shape)):<20} {layer3_params:,}")
    
    # Layer 4 (3 blocks)
    x = model.layer4(x)
    layer4_params = sum(p.numel() for p in model.layer4.parameters())
    print(f"{'Layer4 (3 ResBlocks)':<30} {'Sequential':<20} {str(tuple(x.shape)):<20} {layer4_params:,}")
    
    # Global Average Pooling
    x = model.global_avg_pool(x)
    print(f"{'Global Average Pool':<30} {'AdaptiveAvgPool2d':<20} {str(tuple(x.shape)):<20} {'0'}")
    
    # Flatten
    x = model.flatten(x)
    print(f"{'Flatten':<30} {'Flatten':<20} {str(tuple(x.shape)):<20} {'0'}")
    
    # Dropout
    x = model.dropout(x)
    print(f"{'Dropout (p=0.3)':<30} {'Dropout':<20} {str(tuple(x.shape)):<20} {'0'}")
    
    # Fully Connected
    x = model.fc(x)
    fc_params = sum(p.numel() for p in model.fc.parameters())
    print(f"{'Fully Connected':<30} {'Linear':<20} {str(tuple(x.shape)):<20} {fc_params:,}")
    
    print("-" * 100)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL PARAMETERS':<30} {'':<20} {'':<20} {total:,}")
    print("=" * 100)
    
    return stem_params, layer1_params, layer2_params, layer3_params, layer4_params, fc_params


def visualize_full_architecture():
    """Create visual diagram of the full CNN architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.axis('off')
    
    # Define layer structure
    layers = [
        ("Input\n3×224×224", 0, "lightblue", 3),
        ("Conv 7×7, 64\nstride=2", 1, "orange", 64),
        ("Batch Norm", 1.5, "lightgreen", 64),
        ("ReLU", 2, "yellow", 64),
        ("MaxPool 3×3\nstride=2", 2.5, "orange", 64),
        ("ResBlock × 3\n64 channels", 3.5, "pink", 64),
        ("ResBlock × 4\n128 channels", 5, "pink", 128),
        ("ResBlock × 6\n256 channels", 7, "pink", 256),
        ("ResBlock × 3\n512 channels", 9.5, "pink", 512),
        ("Global Avg Pool\n512×1×1", 11, "lightblue", 512),
        ("Dropout (0.3)", 11.5, "lightgreen", 512),
        ("FC Layer\n512→9", 12.5, "orange", 9),
        ("Output\n9 classes", 13.5, "lightcoral", 9)
    ]
    
    # Draw layers
    for layer_name, y_pos, color, channels in layers:
        width = 0.5 + (channels / 512) * 1.5  # Width based on channels
        rect = plt.Rectangle((2 - width/2, y_pos), width, 0.4, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(2, y_pos + 0.2, layer_name, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        # Add channel info
        if channels > 9:
            ax.text(3.2, y_pos + 0.2, f'{channels}ch', ha='left', va='center', 
                   fontsize=8, style='italic', color='darkblue')
    
    # Draw connections
    for i in range(len(layers) - 1):
        y1 = layers[i][1] + 0.4
        y2 = layers[i+1][1]
        ax.plot([2, 2], [y1, y2], 'k-', linewidth=2)
        ax.arrow(2, y2-0.05, 0, 0.05, head_width=0.15, head_length=0.05, fc='black', ec='black')
    
    # Add residual connection indicators
    residual_blocks = [(3.5, "3 blocks"), (5, "4 blocks"), (7, "6 blocks"), (9.5, "3 blocks")]
    for y_pos, label in residual_blocks:
        arc = plt.Circle((2, y_pos + 0.2), 0.6, fill=False, 
                        edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(arc)
        ax.text(3.5, y_pos + 0.2, f'Skip\nConnections', ha='left', va='center',
               fontsize=7, color='red', style='italic')
    
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 14.5)
    ax.invert_yaxis()
    ax.set_title('Deeper ResNet-34 CNN Architecture\n[3, 4, 6, 3] Residual Blocks with Channel Attention', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('deeper_resnet34_architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Architecture diagram saved as 'deeper_resnet34_architecture_diagram.png'")
    plt.close()


def visualize_block_details():
    """Create detailed visualization of Residual Block and Channel Attention."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Residual Block Structure
    ax = axes[0]
    ax.axis('off')
    ax.set_title('Residual Block with Channel Attention', fontsize=12, fontweight='bold')
    
    block_layers = [
        ("Input (x)", 0, "lightblue"),
        ("Conv 3×3", 1, "orange"),
        ("Batch Norm", 1.7, "lightgreen"),
        ("ReLU", 2.4, "yellow"),
        ("Conv 3×3", 3.1, "orange"),
        ("Batch Norm", 3.8, "lightgreen"),
        ("Channel Attention\n(SE Block)", 4.5, "pink"),
        ("Add (x + residual)", 5.5, "lightcoral"),
        ("ReLU", 6.2, "yellow"),
        ("Output", 7, "lightblue")
    ]
    
    for layer_name, y_pos, color in block_layers:
        rect = plt.Rectangle((1, y_pos), 2, 0.5, facecolor=color, 
                            edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(2, y_pos + 0.25, layer_name, ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Draw main path connections
    for i in range(len(block_layers) - 1):
        y1 = block_layers[i][1] + 0.5
        y2 = block_layers[i+1][1]
        ax.plot([2, 2], [y1, y2], 'k-', linewidth=2)
        ax.arrow(2, y2-0.05, 0, 0.05, head_width=0.2, head_length=0.05, 
                fc='black', ec='black')
    
    # Draw skip connection
    ax.plot([3.5, 3.5, 3.5], [0.25, 5.75, 5.75], 'r--', linewidth=3, label='Skip Connection')
    ax.arrow(3.5, 5.75, -1.3, 0, head_width=0.15, head_length=0.15, fc='red', ec='red')
    ax.legend(loc='upper right')
    
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.5, 7.5)
    ax.invert_yaxis()
    
    # Channel Attention Detail
    ax = axes[1]
    ax.axis('off')
    ax.set_title('Channel Attention (Squeeze-Excitation) Block', fontsize=12, fontweight='bold')
    
    se_layers = [
        ("Input\nFeature Maps", 0, "lightblue"),
        ("Global Avg Pool\n(Squeeze)", 1, "orange"),
        ("FC → ReLU\n(Excitation)", 2, "lightgreen"),
        ("FC → Sigmoid", 3, "yellow"),
        ("Channel Weights", 3.7, "pink"),
        ("Multiply\n(Recalibration)", 4.7, "lightcoral"),
        ("Output\nRecalibrated Maps", 5.7, "lightblue")
    ]
    
    for layer_name, y_pos, color in se_layers:
        rect = plt.Rectangle((1, y_pos), 2, 0.5, facecolor=color, 
                            edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(2, y_pos + 0.25, layer_name, ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Draw connections
    for i in range(len(se_layers) - 1):
        y1 = se_layers[i][1] + 0.5
        y2 = se_layers[i+1][1]
        ax.plot([2, 2], [y1, y2], 'k-', linewidth=2)
        ax.arrow(2, y2-0.05, 0, 0.05, head_width=0.2, head_length=0.05, 
                fc='black', ec='black')
    
    # Draw feature map path
    ax.plot([3.5, 3.5], [0.25, 4.95], 'b--', linewidth=2, label='Feature Maps Path')
    ax.arrow(3.5, 4.95, -1.3, 0, head_width=0.15, head_length=0.15, fc='blue', ec='blue')
    ax.legend(loc='upper right')
    
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.5, 6.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('deeper_resnet34_block_details.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Block details saved as 'deeper_resnet34_block_details.png'")
    plt.close()


def visualize_parameter_distribution(param_counts):
    """Create parameter distribution charts."""
    stem_params, layer1_params, layer2_params, layer3_params, layer4_params, fc_params = param_counts
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of parameters by layer group
    ax = axes[0]
    counts = [stem_params, layer1_params, layer2_params, 
             layer3_params, layer4_params, fc_params]
    labels = ['Stem', 'Layer1\n(3 blocks)', 'Layer2\n(4 blocks)', 
             'Layer3\n(6 blocks)', 'Layer4\n(3 blocks)', 'FC Layer']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    
    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Parameter Distribution by Layer Group', fontweight='bold')
    
    # Bar chart of parameters
    ax = axes[1]
    ax.bar(range(len(labels)), counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Parameters per Layer Group', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts)*0.02, f'{v:,}', 
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('deeper_resnet34_parameter_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Parameter distribution saved as 'deeper_resnet34_parameter_distribution.png'")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to generate all visualizations."""
    print("\n" + "=" * 100)
    print("🎨 CNN ARCHITECTURE VISUALIZATION TOOL")
    print("=" * 100)
    print("\nGenerating comprehensive visualizations for Deeper ResNet-34 CNN...")
    print()
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeeperResidualCNN(num_classes=9).to(device)
    
    print(f"✅ Model created on device: {device}")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 1. Print text-based summary
    param_counts = print_architecture_summary(model, device)
    
    # 2. Create full architecture diagram
    print("\n📊 Creating full architecture diagram...")
    visualize_full_architecture()
    
    # 3. Create block details visualization
    print("📊 Creating block details visualization...")
    visualize_block_details()
    
    # 4. Create parameter distribution charts
    print("📊 Creating parameter distribution charts...")
    visualize_parameter_distribution(param_counts)
    
    print("\n" + "=" * 100)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 100)
    print("\n📁 Generated files:")
    print("   • deeper_resnet34_architecture_diagram.png")
    print("   • deeper_resnet34_block_details.png")
    print("   • deeper_resnet34_parameter_distribution.png")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
