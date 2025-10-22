import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from setup import DEVICE, NUM_CLASSES
from data_loader import get_data_loaders
from model import ImprovedCNN

def evaluate_model():
    """
    Evaluate the best trained model on the test set.
    Generate confusion matrix, classification report, and accuracy metrics.
    """
    print("Loading test data...")
    _, _, test_loader, class_names = get_data_loaders()
    
    print("Loading best model...")
    model = ImprovedCNN(num_classes=len(class_names)).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Best model loaded successfully!")
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please train the model first.")
        return
    
    model.eval()
    
    print("Evaluating model on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred = torch.cat(all_preds).cpu().numpy()
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report to file
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report - Test Set Evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)
    
    print("Classification report saved to 'classification_report.txt'")
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i, class_name in enumerate(class_names):
        class_correct = np.sum((y_true == i) & (y_pred == i))
        class_total = np.sum(y_true == i)
        if class_total > 0:
            class_acc = class_correct / class_total
            print(f"{class_name}: {class_acc:.4f} ({class_correct}/{class_total})")
        else:
            print(f"{class_name}: No samples in test set")
    
    # Additional metrics
    print(f"\nAdditional Metrics:")
    print(f"Total test samples: {len(y_true)}")
    print(f"Correct predictions: {np.sum(y_true == y_pred)}")
    print(f"Incorrect predictions: {np.sum(y_true != y_pred)}")
    
    return test_accuracy, report, cm

def plot_class_distribution(y_true, class_names):
    """Plot the distribution of classes in the test set."""
    unique, counts = np.unique(y_true, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, color='skyblue', edgecolor='navy')
    plt.title('Test Set Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('test_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting model evaluation...")
    print("=" * 50)
    
    # Load test data to get class distribution
    _, _, test_loader, class_names = get_data_loaders()
    
    # Get all test labels for distribution plot
    all_test_labels = []
    for _, labels in test_loader:
        all_test_labels.extend(labels.numpy())
    
    # Plot class distribution
    plot_class_distribution(np.array(all_test_labels), class_names)
    
    # Evaluate model
    test_accuracy, report, cm = evaluate_model()
    
    print("\nEvaluation completed!")
    print("Generated files:")
    print("- confusion_matrix.png")
    print("- classification_report.txt")
    print("- test_class_distribution.png")
