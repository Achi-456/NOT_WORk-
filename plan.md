# Plan: EN3150 Assignment 3, Part 1 (Custom CNN with PyTorch)

[cite_start]**Objective:** Complete all tasks for Part 1 (Questions 4-12) of the assignment [cite: 29-54].
**Dataset:** RealWaste (9 classes).
**Model:** Use the user-provided `build_improved_cnn` architecture, adapted for PyTorch.

---

## Step 1: Project Setup (File: `setup.py`)

**Agent Task:** Create a script to set up the environment and split data.

1.  **Imports:** `os`, `shutil`, `numpy`, `sklearn.model_selection`, `glob`.
2.  **Define Constants:**
    * `IMAGE_SIZE = (224, 224)` (to match the model's input)
    * `BATCH_SIZE = 32`
    * `NUM_CLASSES = 9` (This is critical, as the RealWaste dataset has 9 classes)
    * [cite_start]`EPOCHS = 20` [cite: 42]
    * `DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`
    * `RAW_DATA_DIR = 'RealWaste'`
    * `SPLIT_DATA_DIR = 'data'`
3.  **Dataset Download:** (Manual Step) Assume the `RealWaste` dataset folder is already downloaded and present.
4.  **Data Splitting (Q3):**
    * Create a function `split_data()` that reads the raw `RAW_DATA_DIR`.
    * It must split the images from all 9 subfolders into `data/train`, `data/validation`, and `data/test` directories.
    * [cite_start]**The split *must* be 70% training, 15% validation, and 15% testing**[cite: 9].
    * Use `sklearn.model_selection.train_test_split` twice to achieve this, ensuring `stratify` is used to maintain class distribution.
    * This function will be called once to prepare the directory structure.

---

## Step 2: Data Loaders (File: `data_loader.py`)

**Agent Task:** Create a file to define PyTorch data loaders.

1.  **Imports:** `torch`, `torchvision.datasets`, `torchvision.transforms`, `torch.utils.data.DataLoader`, `setup`.
2.  **Define Transforms:**
    * Define `train_transform`: `transforms.Resize(IMAGE_SIZE)`, `transforms.RandomHorizontalFlip()`, `transforms.RandomRotation(10)`, `transforms.ToTensor()`, `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` (ImageNet stats).
    * Define `val_test_transform`: `transforms.Resize(IMAGE_SIZE)`, `transforms.ToTensor()`, `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.
3.  **Create DataLoaders Function:**
    * Define a function `get_data_loaders()` that:
        * Creates `train_dataset` using `datasets.ImageFolder(root='data/train', transform=train_transform)`.
        * Creates `val_dataset` using `datasets.ImageFolder(root='data/validation', transform=val_test_transform)`.
        * Creates `test_dataset` using `datasets.ImageFolder(root='data/test', transform=val_test_transform)`.
        * Wraps them in `DataLoader` with `batch_size=BATCH_SIZE`.
        * Returns `train_loader`, `val_loader`, `test_loader`, and `train_dataset.classes`.

---

## Step 3: Model Definition (File: `model.py`)

**Agent Task:** Create a file that defines the CNN architecture in PyTorch.

1.  **Imports:** `torch`, `torch.nn`, `torch.nn.functional as F`.
2.  **ChannelAttention (SE Block):**
    * Define a class `ChannelAttention(nn.Module)` that replicates the Squeeze-and-Excitation block.
    * It will use `nn.AdaptiveAvgPool2d`, two `nn.Conv2d` (as 1x1 convolutions), `nn.ReLU`, and `nn.Sigmoid`.
3.  **ImprovedCNN (Q4, Q5):**
    * Define a class `ImprovedCNN(nn.Module)` that inherits from `nn.Module`.
    * In `__init__`, define all the layers as specified in the Keras code, translating them to PyTorch:
        * **Block 1:** `nn.Conv2d(3, 64)`, `nn.BatchNorm2d(64)`, `nn.ReLU()`, `ChannelAttention(64)`, `nn.MaxPool2d(2)`, `nn.Dropout2d(0.25)` (PyTorch's `Dropout2d` is spatial dropout).
        * **Block 2:** `nn.Conv2d(64, 128)`, `nn.BatchNorm2d(128)`, `nn.ReLU()`, `ChannelAttention(128)`, `nn.MaxPool2d(2)`, `nn.Dropout2d(0.3)`.
        * **Block 3:** `nn.Conv2d(128, 256)`, `nn.BatchNorm2d(256)`, `nn.ReLU()`, `ChannelAttention(256)`, `nn.MaxPool2d(2)`, `nn.Dropout2d(0.3)`.
        * **Classifier Head:** `nn.AdaptiveAvgPool2d((1, 1))` (for global pooling), `nn.Flatten()`, `nn.Linear(256, 512)`, `nn.BatchNorm1d(512)`, `nn.ReLU()`, `nn.Dropout(0.5)`.
        * **Output Layer:** `nn.Linear(512, NUM_CLASSES)` (**CRITICAL FIX**: must be 9 classes).
    * In `__init__`, apply `nn.init.kaiming_normal_` (He normal) to all `nn.Conv2d` and `nn.Linear` layers.
    * Define the `forward(self, x)` method, passing `x` through all the layers in sequence.

---

## Step 4: Training & Comparison (File: `train_compare.py`)

**Agent Task:** Create a script to train models with the three optimizers for Question 10.

1.  **Imports:** `setup`, `data_loader`, `model`, `torch`, `torch.nn`, `torch.optim`, `matplotlib.pyplot`, `time`.
2.  **Define Train/Val Loops:**
    * `train_one_epoch(model, loader, criterion, optimizer, device)`: loops over data, performs forward/backward pass, updates weights, returns epoch loss and accuracy.
    * `validate(model, loader, criterion, device)`: loops over data with `torch.no_grad()`, returns epoch loss and accuracy.
3.  **Define Full Training Function:**
    * `train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_name)`:
        * Iterates from `1` to `epochs`.
        * Calls `train_one_epoch` and `validate`.
        * Prints stats for each epoch.
        * Saves train/val loss/acc to a `history` dictionary.
        * If it's the `Adam` model, saves the best weights to `best_model.pth`.
        * Returns the `history` dictionary.
4.  **Main Execution:**
    * Load data: `train_loader, val_loader, _, class_names = get_data_loaders()`.
    * `criterion = nn.CrossEntropyLoss()`.
    * **Define Models & Optimizers (Q10):**
        * **Model 1 (Adam):**
            * `model_adam = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)`
            * [cite_start]`optimizer_adam = optim.Adam(model_adam.parameters())` [cite: 43]
        * **Model 2 (SGD):**
            * `model_sgd = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)`
            * [cite_start]`optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)` [cite: 45]
        * **Model 3 (SGD+Momentum):**
            * `model_momentum = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)`
            * [cite_start]`optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)` [cite: 45]
    * **Train Models (Q7, Q10):**
        * `history_adam = train_model(model_adam, ..., 'Adam')`
        * `history_sgd = train_model(model_sgd, ..., 'SGD')`
        * `history_momentum = train_model(model_momentum, ..., 'SGD+Momentum')`
    * **Generate Plots (Q7, Q10):**
        * [cite_start]Create `optimizer_loss_comparison.png`: Plots Training Loss and Validation Loss for all 3 histories[cite: 42].
        * Create `optimizer_accuracy_comparison.png`: Plots Training Accuracy and Validation Accuracy for all 3 histories.

---

## Step 5: Model Evaluation (File: `evaluate.py`)

**Agent Task:** Create a script to evaluate the best model on the test set for Question 12.

1.  **Imports:** `setup`, `data_loader`, `model`, `torch`, `sklearn.metrics`, `seaborn`, `matplotlib.pyplot`, `numpy`.
2.  **Main Execution:**
    * Load data: `_, _, test_loader, class_names = get_data_loaders()`.
    * **Load Model:**
        * `model = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)`
        * `model.load_state_dict(torch.load('best_model.pth'))`
        * `model.eval()`
    * **Evaluate (Q12):**
        * Initialize `all_preds = []` and `all_labels = []`.
        * Loop through `test_loader` with `torch.no_grad()`.
        * Get model `outputs`.
        * Get `preds = torch.argmax(outputs, 1)`.
        * Append `preds` and `labels` to the lists.
        * Concatenate all tensors and move to CPU: `y_true = torch.cat(all_labels).cpu()`, `y_pred = torch.cat(all_preds).cpu()`.
    * **Generate Outputs (Q12):**
        * [cite_start]**Test Accuracy:** Calculate `accuracy_score(y_true, y_pred)` and print it[cite: 54].
        * **Confusion Matrix:**
            * Compute `cm = confusion_matrix(y_true, y_pred)`.
            * Plot with `seaborn.heatmap`, using `class_names` as labels.
            * [cite_start]Save as `confusion_matrix.png`[cite: 54].
        * **Precision/Recall:**
            * Generate `report = classification_report(y_true, y_pred, target_names=class_names)`.
            * [cite_start]Save `report` to `classification_report.txt`[cite: 54].

---

## Step 6: Report Generation (File: `generate_report.py`)

**Agent Task:** Create a script that prints all text-based answers for the final report.

1.  **Imports:** `model`, `torchsummary` (requires `pip install torchsummary`).
2.  **Q4/Q5 (Model Architecture):**
    * `model = ImprovedCNN(num_classes=9)`
    * Use `torchsummary.summary(model, (3, 224, 224))` to get a Keras-like summary.
    * Save this summary to `model_summary.txt`.
3.  **Q6 (Activation Functions):**
    * Print a markdown string justifying:
        * **ReLU:** "Used in all hidden layers. It's computationally efficient, avoids the vanishing gradient problem, and introduces non-linearity."
        * [cite_start]**Softmax:** "A Softmax function is implicitly applied by the `nn.CrossEntropyLoss` criterion during training. For the final output layer, it converts the model's raw logits into a probability distribution across all 9 classes, which is essential for multi-class classification." [cite: 37]
4.  **Q8 (Optimizer Choice):**
    * [cite_start]Print justification for **Adam**: "We chose the **Adam** optimizer[cite: 43]. It combines the benefits of adaptive learning rates (like RMSprop) and momentum. This allows for faster convergence and makes the model less sensitive to the initial learning rate choice compared to standard SGD."
5.  **Q9 (Learning Rate):**
    * Print the answer: "For Adam, we used the default learning rate (0.001). For SGD and SGD+Momentum, a learning rate of 0.01 was selected as a standard starting point for those optimizers."
6.  **Q10 (Performance Metrics):**
    * [cite_start]Print the answer: "We used **Validation Accuracy** and **Validation Loss** to compare optimizers[cite: 46]. These metrics are crucial as they measure the model's performance on unseen data (the validation set), which is the best indicator of its ability to generalize and avoid overfitting."
7.  **Q11 (Momentum Impact):**
    * Print placeholder: **(Human-in-the-loop):** "Based on the `optimizer_accuracy_comparison.png` plot, the SGD model with momentum... [Agent: Describe the difference between the SGD and SGD+Momentum lines. Did it converge faster? Was the final accuracy higher? [cite_start]Was the curve smoother?]" [cite: 49]
8.  **Q12 (Evaluation Summary):**
    * Print summary: "The final model was evaluated on the 15% testing dataset. The results are as follows:
        * [cite_start]**Test Accuracy:** [Read from `evaluate.py` output] [cite: 54]
        * [cite_start]**Precision/Recall/F1-Score:** [See `classification_report.txt`] [cite: 54]
        * [cite_start]**Confusion Matrix:** [See `confusion_matrix.png`] [cite: 54]"