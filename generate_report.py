import torch
from model import ImprovedCNN
from setup import NUM_CLASSES

def generate_model_summary():
    """Generate and save model architecture summary."""
    try:
        from torchsummary import summary
        torchsummary_available = True
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        torchsummary_available = False
    
    # Create model
    model = ImprovedCNN(num_classes=NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("Model Architecture Summary")
    print("=" * 80)
    
    if torchsummary_available:
        try:
            # Generate torchsummary
            summary(model, (3, 224, 224))
        except Exception as e:
            print(f"Error generating torchsummary: {e}")
            print("Falling back to manual summary...")
            print_manual_summary(model)
    else:
        print_manual_summary(model)
    
    # Save summary to file
    try:
        if torchsummary_available:
            import io
            import sys
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                summary(model, (3, 224, 224))
            summary_str = f.getvalue()
            
            with open('model_summary.txt', 'w') as file:
                file.write("Model Architecture Summary\n")
                file.write("=" * 80 + "\n")
                file.write(summary_str)
        else:
            with open('model_summary.txt', 'w') as file:
                file.write("Model Architecture Summary\n")
                file.write("=" * 80 + "\n")
                file.write(get_manual_summary_str(model))
    except Exception as e:
        print(f"Error saving summary to file: {e}")

def print_manual_summary(model):
    """Print manual model summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("\nModel Layers:")
    print("-" * 50)
    
    for name, module in model.named_children():
        print(f"{name}: {module}")

def get_manual_summary_str(model):
    """Get manual model summary as string."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary_str = f"Total Parameters: {total_params:,}\n"
    summary_str += f"Trainable Parameters: {trainable_params:,}\n\n"
    summary_str += "Model Layers:\n"
    summary_str += "-" * 50 + "\n"
    
    for name, module in model.named_children():
        summary_str += f"{name}: {module}\n"
    
    return summary_str

def generate_answers():
    """Generate answers for all assignment questions."""
    
    print("\n" + "=" * 80)
    print("EN3150 ASSIGNMENT 3 - PART 1 ANSWERS")
    print("=" * 80)
    
    # Q4/Q5: Model Architecture
    print("\n**Q4 & Q5: MODEL ARCHITECTURE**")
    print("-" * 50)
    generate_model_summary()
    
    # Q6: Activation Functions
    print("\n**Q6: ACTIVATION FUNCTIONS JUSTIFICATION**")
    print("-" * 50)
    activation_justification = """
**ReLU Activation Function:**
- Used in all hidden layers (convolutional and fully connected layers)
- Computationally efficient with simple thresholding operation
- Helps mitigate the vanishing gradient problem
- Introduces non-linearity essential for learning complex patterns
- Sparse activation (outputs 0 for negative inputs) leads to efficient computation

**Softmax Activation Function:**
- Applied implicitly through nn.CrossEntropyLoss during training
- For inference, softmax converts raw logits to probability distribution
- Essential for multi-class classification with 9 classes
- Ensures output probabilities sum to 1
- Provides interpretable confidence scores for each class prediction
"""
    print(activation_justification)
    
    # Q8: Optimizer Choice
    print("\n**Q8: OPTIMIZER CHOICE JUSTIFICATION**")
    print("-" * 50)
    optimizer_justification = """
**Adam Optimizer Selection:**

We chose Adam (Adaptive Moment Estimation) as our primary optimizer for the following reasons:

1. **Adaptive Learning Rates**: Adam adapts learning rates for each parameter individually,
   making it robust to different parameter scales and gradients.

2. **Momentum Integration**: Combines benefits of momentum (RMSprop) with bias correction,
   leading to faster convergence and better generalization.

3. **Reduced Sensitivity**: Less sensitive to initial learning rate choice compared to SGD,
   making it more user-friendly and reliable.

4. **Efficient Computation**: Computationally efficient and requires minimal memory overhead.

5. **Empirical Performance**: Consistently performs well across various deep learning tasks
   and architectures, making it a reliable default choice.
"""
    print(optimizer_justification)
    
    # Q9: Learning Rate
    print("\n**Q9: LEARNING RATE SELECTION**")
    print("-" * 50)
    learning_rate_explanation = """
**Learning Rate Configuration:**

- **Adam Optimizer**: Uses default learning rate of 0.001
  - Adam's adaptive nature makes it less sensitive to learning rate choice
  - Default rate of 0.001 is empirically proven to work well across many tasks
  - Adam automatically adjusts effective learning rates during training

- **SGD and SGD+Momentum**: Uses learning rate of 0.01
  - Higher learning rate needed for SGD due to lack of adaptive scaling
  - Standard starting point for SGD-based optimizers
  - Momentum helps stabilize training with this learning rate
"""
    print(learning_rate_explanation)
    
    # Q10: Performance Metrics
    print("\n**Q10: PERFORMANCE METRICS FOR OPTIMIZER COMPARISON**")
    print("-" * 50)
    performance_metrics = """
**Selected Metrics: Validation Accuracy and Validation Loss**

**Justification:**

1. **Validation Accuracy**: 
   - Primary metric for classification tasks
   - Directly interpretable (percentage of correct predictions)
   - Measures model's ability to generalize to unseen data
   - Avoids overfitting bias present in training accuracy

2. **Validation Loss**:
   - Provides smooth, continuous metric for optimization comparison
   - More sensitive to small performance changes than accuracy
   - Helps identify convergence behavior and training stability
   - CrossEntropyLoss naturally handles multi-class probability distributions

**Why Validation over Training Metrics:**
- Training metrics can be misleading due to overfitting
- Validation metrics better represent real-world performance
- Essential for fair comparison between different optimizers
- Helps in model selection and hyperparameter tuning
"""
    print(performance_metrics)
    
    # Q11: Momentum Impact (Placeholder)
    print("\n**Q11: MOMENTUM IMPACT ANALYSIS**")
    print("-" * 50)
    momentum_analysis = """
**Analysis Based on optimizer_accuracy_comparison.png:**

[To be filled after training completion - requires human analysis of the generated plots]

Expected observations:
- SGD with momentum should show smoother convergence curves
- Reduced oscillations in loss compared to vanilla SGD
- Potentially faster convergence to optimal solution
- Better final validation accuracy due to improved optimization trajectory

The momentum term helps SGD by:
1. Accelerating convergence in consistent gradient directions
2. Dampening oscillations in areas with high curvature
3. Helping escape local minima through accumulated momentum
4. Providing more stable training dynamics
"""
    print(momentum_analysis)
    
    # Q12: Evaluation Summary (Placeholder)
    print("\n**Q12: MODEL EVALUATION SUMMARY**")
    print("-" * 50)
    evaluation_summary = """
**Final Model Evaluation Results:**

The best performing model (Adam optimizer) was evaluated on the 15% test dataset:

**Test Accuracy**: [To be filled from evaluate.py output]

**Detailed Metrics**: 
- See classification_report.txt for precision, recall, and F1-scores per class
- Confusion matrix visualization available in confusion_matrix.png
- Per-class performance analysis included in evaluation output

**Evaluation Methodology**:
- Unbiased test set (never seen during training/validation)
- Comprehensive metrics including precision, recall, F1-score
- Class-wise performance analysis for identifying model strengths/weaknesses
- Visual confusion matrix for error pattern analysis
"""
    print(evaluation_summary)
    
    # Save all answers to file
    save_answers_to_file(activation_justification, optimizer_justification, 
                        learning_rate_explanation, performance_metrics, 
                        momentum_analysis, evaluation_summary)

def save_answers_to_file(activation_just, optimizer_just, lr_explanation, 
                        perf_metrics, momentum_analysis, eval_summary):
    """Save all answers to a comprehensive report file."""
    
    with open('assignment_answers.txt', 'w') as f:
        f.write("EN3150 ASSIGNMENT 3 - PART 1 COMPREHENSIVE ANSWERS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Q4 & Q5: MODEL ARCHITECTURE\n")
        f.write("-" * 50 + "\n")
        f.write("See model_summary.txt for detailed architecture.\n\n")
        
        f.write("Q6: ACTIVATION FUNCTIONS JUSTIFICATION\n")
        f.write("-" * 50 + "\n")
        f.write(activation_just + "\n\n")
        
        f.write("Q8: OPTIMIZER CHOICE JUSTIFICATION\n")
        f.write("-" * 50 + "\n")
        f.write(optimizer_just + "\n\n")
        
        f.write("Q9: LEARNING RATE SELECTION\n")
        f.write("-" * 50 + "\n")
        f.write(lr_explanation + "\n\n")
        
        f.write("Q10: PERFORMANCE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(perf_metrics + "\n\n")
        
        f.write("Q11: MOMENTUM IMPACT ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(momentum_analysis + "\n\n")
        
        f.write("Q12: MODEL EVALUATION SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(eval_summary + "\n\n")
    
    print("\nAll answers saved to 'assignment_answers.txt'")

if __name__ == "__main__":
    generate_answers()
