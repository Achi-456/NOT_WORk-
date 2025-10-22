# Learning Rate Scheduler Integration Guide

## 🎯 What's Missing: Learning Rate Scheduling

Your current notebook does NOT have learning rate scheduling. This is an important feature that can improve training performance.

## ✅ How to Add It

### Step 1: Update `train_model()` Function Signature

**Current:**
```python
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_name):
```

**Updated:**
```python
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_name, use_scheduler=True):
```

### Step 2: Add Scheduler Initialization

Add this code RIGHT AFTER the `history` and `best_val_acc` initialization (around line 890):

```python
# Initialize learning rate scheduler
scheduler = None
if use_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Monitor validation loss (minimize)
        factor=0.5,           # Reduce LR by factor of 0.5
        patience=3,           # Wait 3 epochs before reducing
        verbose=True,         # Print messages
        min_lr=1e-6          # Don't go below this LR
    )
    print(f"📈 Learning Rate Scheduler: ReduceLROnPlateau")
    print(f"   • Factor: 0.5 | Patience: 3 epochs | Min LR: 1e-6")

# Also add learning rate tracking to history
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []}
```

### Step 3: Track Learning Rate Each Epoch

Add this code at the BEGINNING of the epoch loop (after `for epoch in range(epochs):`):

```python
# Get current learning rate
current_lr = optimizer.param_groups[0]['lr']
history['learning_rates'].append(current_lr)

print(f"\\n{'─'*50}")
print(f"🔄 Epoch [{epoch+1}/{epochs}] (LR: {current_lr:.6f}):")
```

### Step 4: Step the Scheduler

Add this code AFTER validation (after you have `val_loss`):

```python
# Learning rate scheduling
if scheduler is not None:
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)  # Step based on validation loss
    new_lr = optimizer.param_groups[0]['lr']
    
    if new_lr != old_lr:
        print(f"   🔽 Learning Rate reduced: {old_lr:.6f} → {new_lr:.6f}")
```

### Step 5: Update Epoch Summary Print

Update your epoch summary to include LR:

```python
print(f"📊 EPOCH [{epoch+1}/{epochs}] SUMMARY:")
print(f"   📚 LR    → {current_lr:.6f}")  # ADD THIS LINE
print(f"   🏃 Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
print(f"   🎯 Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
```

### Step 6: Update Final Summary

Add to the final training complete summary:

```python
print(f"🎉 TRAINING COMPLETE - {model_name.upper()}")
print(f"   ⏱️  Total Time: {total_time/60:.1f} min")
print(f"   🏆 Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"   📚 Initial LR: {history['learning_rates'][0]:.6f}")  # ADD THIS
print(f"   📚 Final LR: {history['learning_rates'][-1]:.6f}")   # ADD THIS
```

## 📊 Optional: Add Learning Rate Visualization

Add a new cell after training to visualize LR schedule:

```python
def plot_learning_rate_schedule(all_histories):
    \"\"\"Visualize learning rate schedules for all trained models.\"\"\"
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = {'Adam': 'blue', 'SGD': 'red', 'SGD+Momentum': 'green'}
    
    # Plot 1: Learning Rate over Epochs
    ax = axes[0]
    for name, history in all_histories.items():
        if 'learning_rates' in history and len(history['learning_rates']) > 0:
            epochs = range(1, len(history['learning_rates']) + 1)
            ax.plot(epochs, history['learning_rates'], color=colors[name], 
                   label=name, linewidth=2.5, marker='o', markersize=6)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule - All Optimizers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see changes better
    
    # Plot 2: Learning Rate vs Validation Loss
    ax = axes[1]
    for name, history in all_histories.items():
        if 'learning_rates' in history and 'val_loss' in history:
            epochs = range(1, len(history['learning_rates']) + 1)
            # Plot both LR and val loss on same axes
            ax2 = ax.twinx()
            
            line1 = ax.plot(epochs, history['learning_rates'], color=colors[name], 
                          label=f'{name} LR', linewidth=2, linestyle='--', alpha=0.7)
            line2 = ax2.plot(epochs, history['val_loss'], color=colors[name], 
                           label=f'{name} Val Loss', linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate vs Validation Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PART1_DIR, 'learning_rate_schedules.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\\n" + "="*80)
    print("📚 LEARNING RATE REDUCTION SUMMARY")
    print("="*80)
    for name, history in all_histories.items():
        if 'learning_rates' in history:
            initial_lr = history['learning_rates'][0]
            final_lr = history['learning_rates'][-1]
            reductions = sum(1 for i in range(1, len(history['learning_rates'])) 
                           if history['learning_rates'][i] < history['learning_rates'][i-1])
            
            print(f"\\n{name}:")
            print(f"   Initial LR: {initial_lr:.6f}")
            print(f"   Final LR:   {final_lr:.6f}")
            print(f"   Reductions: {reductions} times")
            if initial_lr > final_lr:
                print(f"   Total reduction: {initial_lr/final_lr:.2f}x")
    print("="*80)

# Call it after training all models
plot_learning_rate_schedule(all_histories)
```

## 🔧 Benefits of Learning Rate Scheduling

1. **Better Convergence**: Starts with higher LR for fast initial learning
2. **Fine-tuning**: Reduces LR when stuck to fine-tune weights
3. **Higher Final Accuracy**: Usually improves final test accuracy by 1-3%
4. **Training Stability**: Prevents oscillations in later epochs
5. **Automatic**: No manual intervention needed

## 📝 Where to Make Changes

1. Find the `train_model` function definition (around line 884)
2. Make the 6 changes listed above
3. Add the visualization function after training completes
4. Re-run all training cells

## ⚡ Quick Test

After making changes, you can test with a simple check:

```python
# Test that scheduler is working
test_model = DeeperResidualCNN(num_classes=NUM_CLASSES).to(DEVICE)
test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
test_criterion = nn.CrossEntropyLoss()

# Train for just 2 epochs to verify scheduler works
test_history, test_acc = train_model(
    test_model, train_loader, val_loader,
    test_optimizer, test_criterion, 2, DEVICE, 'TEST'
)

# Check if LR was tracked
if 'learning_rates' in test_history:
    print("✅ Learning rate scheduling is working!")
    print(f"   LR history: {test_history['learning_rates']}")
else:
    print("❌ Learning rate scheduling not implemented yet")
```

## 🎯 Expected Output

When training, you should see messages like:

```
📈 Learning Rate Scheduler: ReduceLROnPlateau
   • Factor: 0.5 | Patience: 3 epochs | Min LR: 1e-6

🔄 Epoch [1/20] (LR: 0.001000):
   ...training...
   
🔄 Epoch [8/20] (LR: 0.001000):
   ...training...
   🔽 Learning Rate reduced: 0.001000 → 0.000500

🔄 Epoch [14/20] (LR: 0.000500):
   ...training...
   🔽 Learning Rate reduced: 0.000500 → 0.000250
```

This shows the scheduler is working and reducing LR when validation loss plateaus!
