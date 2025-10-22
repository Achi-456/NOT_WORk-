# Results Folder Structure

This folder contains all outputs from EN3150 Assignment 3 experiments.

## 📁 Folder Organization

### `/models`
Contains saved PyTorch model weights (.pth files):
- `best_model_Adam.pth` - Part 1: Custom CNN with Adam optimizer
- `best_model_SGD.pth` - Part 1: Custom CNN with SGD optimizer
- `best_model_SGD+Momentum.pth` - Part 1: Custom CNN with SGD+Momentum
- `best_model_ResNet34_FT.pth` - Part 2: Fine-tuned ResNet-34
- `best_model_EfficientNet_B0_FT.pth` - Part 2: Fine-tuned EfficientNet-B0

### `/visualizations`
Contains all plots and figures (.png files):
- Training curves
- Confusion matrices
- Architecture diagrams
- Optimizer comparisons
- Class-wise performance plots
- Misclassification examples

### `/reports`
Contains text-based reports (.txt files):
- Classification reports for all models
- Performance metrics
- Statistical summaries

### `/part1`
Part 1 specific outputs (Custom CNN):
- Model weights from different optimizers
- Training curves and comparisons
- Architecture visualizations
- Adam optimizer detailed analysis

### `/part2`
Part 2 specific outputs (Transfer Learning):
- Fine-tuned model weights
- Confusion matrices for ResNet-34 and EfficientNet-B0
- Fine-tuning comparison plots
- Test set evaluation results

## 🎯 Usage

All notebook cells will automatically save outputs to these organized folders.
This structure makes it easy to:
- Find specific model checkpoints
- Compare visualizations side-by-side
- Archive and share results
- Version control outputs separately

## 📊 Quick Access

- **Best Model**: Check `/models` for saved weights
- **Performance Plots**: Check `/visualizations` for all charts
- **Detailed Metrics**: Check `/reports` for classification reports
