# Plan: Deeper Residual CNN (ResNet-34 Style)

**Objective:** Increase the model's capacity to combat the underfitting observed in the previous training run (where validation accuracy was > 10% higher than training accuracy). This will be done by increasing the depth of the network.

---

## 1. Architectural Changes

We will change the number of residual blocks in our `AdvancedResidualCNN` to match the structure of a **ResNet-34**. This will give the model more layers to learn complex features.

**Current Model (`AdvancedResidualCNN`):**
* Based on ResNet-18 structure.
* Block configuration: `[2, 2, 2, 2]`

**New Model (`DeeperResidualCNN`):**
* Based on ResNet-34 structure.
* Block configuration: `[3, 4, 6, 3]`

---

## 2. Action Plan

### Step 1: Update `model.py`

1.  **Duplicate** the `AdvancedResidualCNN` class and rename the new class to `DeeperResidualCNN`.
2.  In the `__init__` method of `DeeperResidualCNN`, **modify the `_make_layer` calls** to increase the number of blocks:

    ```python
    # --- INSIDE DeeperResidualCNN __init__ ---
    
    # ... (stem is the same) ...
    
    # Residual layers (CHANGED to [3, 4, 6, 3] blocks)
    self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
    self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
    self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
    self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
    
    # ... (classifier and _init_weights are the same) ...
    ```

3.  Ensure the `ResidualBlock` and `ChannelAttention` classes remain unchanged, as they are used by the new model.

### Step 2: Update Training Cell (Jupyter Notebook)

1.  **Import the new model:** Change the import to include `DeeperResidualCNN`.
2.  **Instantiate the new models:** In the "Run the Optimizer Comparison" cell, change the model instantiation for all three optimizers to use `DeeperResidualCNN`:

    ```python
    # --- 1. Adam ---
    print("Training with Adam...")
    model_adam = DeeperResidualCNN(num_classes=NUM_CLASSES).to(DEVICE) # <-- Use new model
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.0001, weight_decay=L2_REG) # <-- Keep LR low for first run
    model_adam, history_adam = train_model(...)
    
    # --- 2. Standard SGD ---
    print("\n" + "="*80 + "\nTraining with Standard SGD...\n")
    model_sgd = DeeperResidualCNN(num_classes=NUM_CLASSES).to(DEVICE) # <-- Use new model
    optimizer_sgd = optim.SGD(...)
    
    # --- 3. SGD with Momentum ---
    print("\n" + "="*80 + "\nTraining with SGD + Momentum...\n")
    model_momentum = DeeperResidualCNN(num_classes=NUM_CLASSES).to(DEVICE) # <-- Use new model
    optimizer_momentum = optim.SGD(...)
    ```

### Step 3: Update Model Summary Cell (Jupyter Notebook)

1.  **Summarize the new model:** Change the model check to use `DeeperResidualCNN` so the report reflects the new, deeper architecture and larger parameter count.

    ```python
    model_check = DeeperResidualCNN(num_classes=NUM_CLASSES).to(DEVICE) # <-- Use new model
    print("=" * 80)
    print(" DEEPER RESIDUAL CNN (ResNet-34 style) ".center(80, '='))
    print("=" * 80)
    summary(model_check, (3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    ...
    ```

### Step 4: Update Evaluation Cell (Jupyter Notebook)

1.  **Load the new model:** Ensure the evaluation step also loads the new architecture before loading the saved `best_model.pth` weights.

    ```python
    model = DeeperResidualCNN(num_classes=NUM_CLASSES).to(DEVICE) # <-- Use new model
    model.load_state_dict(torch.load('best_model.pth'))
    ...
    ```