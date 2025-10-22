import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
from setup import DEVICE, EPOCHS
from data_loader import get_data_loaders
from model import ImprovedCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Average accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate(model, loader, criterion, device):
    """
    Validate the model.
    
    Returns:
        epoch_loss: Average validation loss
        epoch_acc: Average validation accuracy
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_name):
    """
    Train the model for specified epochs and track history.
    
    Returns:
        history: Dictionary containing training history
    """
    print(f"\nTraining {model_name} for {epochs} epochs...")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model (only for Adam optimizer)
        if model_name == 'Adam' and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch [{epoch+1}/{epochs}] - {epoch_time:.2f}s')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    total_time = time.time() - start_time
    print(f'{model_name} training completed in {total_time:.2f}s')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    return history

def plot_comparison(history_adam, history_sgd, history_momentum):
    """
    Generate comparison plots for all three optimizers.
    """
    epochs_range = range(1, len(history_adam['train_loss']) + 1)
    
    # Plot 1: Training and Validation Loss Comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_adam['train_loss'], 'b-', label='Adam Train', linewidth=2)
    plt.plot(epochs_range, history_adam['val_loss'], 'b--', label='Adam Val', linewidth=2)
    plt.plot(epochs_range, history_sgd['train_loss'], 'r-', label='SGD Train', linewidth=2)
    plt.plot(epochs_range, history_sgd['val_loss'], 'r--', label='SGD Val', linewidth=2)
    plt.plot(epochs_range, history_momentum['train_loss'], 'g-', label='SGD+Momentum Train', linewidth=2)
    plt.plot(epochs_range, history_momentum['val_loss'], 'g--', label='SGD+Momentum Val', linewidth=2)
    
    plt.title('Optimizer Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Training and Validation Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_adam['train_acc'], 'b-', label='Adam Train', linewidth=2)
    plt.plot(epochs_range, history_adam['val_acc'], 'b--', label='Adam Val', linewidth=2)
    plt.plot(epochs_range, history_sgd['train_acc'], 'r-', label='SGD Train', linewidth=2)
    plt.plot(epochs_range, history_sgd['val_acc'], 'r--', label='SGD Val', linewidth=2)
    plt.plot(epochs_range, history_momentum['train_acc'], 'g-', label='SGD+Momentum Train', linewidth=2)
    plt.plot(epochs_range, history_momentum['val_acc'], 'g--', label='SGD+Momentum Val', linewidth=2)
    
    plt.title('Optimizer Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Separate plots for clarity
    # Loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_adam['val_loss'], 'b-', label='Adam', linewidth=2, marker='o')
    plt.plot(epochs_range, history_sgd['val_loss'], 'r-', label='SGD', linewidth=2, marker='s')
    plt.plot(epochs_range, history_momentum['val_loss'], 'g-', label='SGD+Momentum', linewidth=2, marker='^')
    
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimizer_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_adam['val_acc'], 'b-', label='Adam', linewidth=2, marker='o')
    plt.plot(epochs_range, history_sgd['val_acc'], 'r-', label='SGD', linewidth=2, marker='s')
    plt.plot(epochs_range, history_momentum['val_acc'], 'g-', label='SGD+Momentum', linewidth=2, marker='^')
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimizer_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function."""
    print("Loading data...")
    train_loader, val_loader, _, class_names = get_data_loaders()
    
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Model 1: Adam optimizer
    print("\n" + "="*50)
    print("Training with Adam Optimizer")
    print("="*50)
    model_adam = ImprovedCNN(num_classes=len(class_names)).to(DEVICE)
    optimizer_adam = optim.Adam(model_adam.parameters())
    history_adam = train_model(model_adam, train_loader, val_loader, 
                              optimizer_adam, criterion, EPOCHS, DEVICE, 'Adam')
    
    # Model 2: SGD optimizer
    print("\n" + "="*50)
    print("Training with SGD Optimizer")
    print("="*50)
    model_sgd = ImprovedCNN(num_classes=len(class_names)).to(DEVICE)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
    history_sgd = train_model(model_sgd, train_loader, val_loader,
                             optimizer_sgd, criterion, EPOCHS, DEVICE, 'SGD')
    
    # Model 3: SGD with Momentum
    print("\n" + "="*50)
    print("Training with SGD + Momentum Optimizer")
    print("="*50)
    model_momentum = ImprovedCNN(num_classes=len(class_names)).to(DEVICE)
    optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)
    history_momentum = train_model(model_momentum, train_loader, val_loader,
                                  optimizer_momentum, criterion, EPOCHS, DEVICE, 'SGD+Momentum')
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(history_adam, history_sgd, history_momentum)
    
    # Print final results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Adam - Final Val Accuracy: {history_adam['val_acc'][-1]:.4f}")
    print(f"SGD - Final Val Accuracy: {history_sgd['val_acc'][-1]:.4f}")
    print(f"SGD+Momentum - Final Val Accuracy: {history_momentum['val_acc'][-1]:.4f}")
    print("="*60)
    
    return history_adam, history_sgd, history_momentum

if __name__ == "__main__":
    history_adam, history_sgd, history_momentum = main()
