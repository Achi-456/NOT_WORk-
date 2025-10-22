# Plan: EN3150 Assignment 3, Part 2 (Transfer Learning)

[cite_start]**Objective:** Complete questions 13-19 by fine-tuning two state-of-the-art pre-trained models, evaluating them, and comparing them to your custom model from Part 1 [cite: 57-66].

**Models (Q13):**
1.  **ResNet-34:** Chosen because your custom model is a ResNet-style architecture. This provides a direct and powerful comparison to show the value of pre-training.
2.  **EfficientNet-B0:** Chosen as the state-of-the-art model. This will be our attempt to reach the 96% accuracy benchmark we researched.

---

## Step 1: Setup for Part 2

**Task:** Add a new cell to your notebook to import new libraries and re-use existing dataloaders.

1.  **Imports:** Add `torchvision.models` to your imports.
2.  **Constants:** Define a new, shorter epoch count for fine-tuning. `EPOCHS_FT = 10`.
3.  **Data:** We will use the **exact same** `dataloaders` from Part 1 to ensure a fair comparison, as required by the assignment.

---

## Step 2: Model Preparation (Q13, Q14)

**Task:** Create a function that loads a pre-trained model and prepares it for fine-tuning.

1.  **Create `get_pretrained_model(model_name)` function:**
    * **Load Model:** Load the specified model with `IMAGENET1K_V1` pre-trained weights.
    * **Freeze Layers:** Freeze all existing parameters. We only want to train the new, final layer.
        ```python
        for param in model.parameters():
            param.requires_grad = False
        ```
    * **Replace Classifier (Q14):**
        * **For ResNet-34:** Replace `model.fc` with a new `nn.Linear(model.fc.in_features, NUM_CLASSES)`.
        * **For EfficientNet-B0:** Replace `model.classifier[1]` with `nn.Linear(model.classifier[1].in_features, NUM_CLASSES)`.
    * **Return Model:** Send the modified model to the `DEVICE`.

---

## Step 3: Training Loop (Q15, Q16)

**Task:** Train the two new models and save their histories and weights.

1.  **Initialize History:** Create a `ft_histories = {}` dictionary.
2.  **Model 1: ResNet-34**
    * `model_resnet = get_pretrained_model('resnet34')`
    * Create an Adam optimizer. **Important:** Pass *only* the new, unfrozen classifier parameters to the optimizer:
        ```python
        optimizer_resnet = optim.Adam(model_resnet.fc.parameters(), lr=0.001)
        ```
    * Call your existing `train_model()` function from Part 1. Train for `EPOCHS_FT` (10 epochs).
    * Save the results: `ft_histories['ResNet-34'] = history` and `torch.save(model_resnet.state_dict(), 'resnet34_finetuned.pth')`.
3.  **Model 2: EfficientNet-B0**
    * `model_effnet = get_pretrained_model('efficientnet_b0')`
    * Create its optimizer, again only passing the new classifier's parameters:
        ```python
        optimizer_effnet = optim.Adam(model_effnet.classifier[1].parameters(), lr=0.001)
        ```
    * Call `train_model()` for this model.
    * Save the results: `ft_histories['EfficientNet-B0'] = history` and `torch.save(model_effnet.state_dict(), 'efficientnet_b0_finetuned.pth')`.
4.  **Plot Results (Q16):**
    * Adapt your plotting function from Part 1 to plot the training/validation loss and accuracy for `ResNet-34` and `EfficientNet-B0` from the `ft_histories` dictionary. Save as `finetune_comparison.png`.

---

## Step 4: Evaluation (Q17)

**Task:** Evaluate the two fine-tuned models on the unseen **test set**.

1.  **Loop and Evaluate:** Create a loop that iterates through your two saved models (`['resnet34', 'efficientnet_b0']`).
2.  **Inside the loop:**
    * Load the correct model architecture: `model = get_pretrained_model(model_name)`.
    * Load the saved weights: `model.load_state_dict(torch.load(f'{model_name}_finetuned.pth'))`.
    * Run the evaluation (using your evaluation code from Part 1) on the `test_loader`.
    * Print the **Test Accuracy**.
    * Generate and save the **Classification Report** (e.g., `classification_report_resnet34.txt`).
    * Generate and save the **Confusion Matrix** (e.g., `confusion_matrix_resnet34.png`).

---

## Step 5: Report & Comparison (Q18, Q19)

**Task:** Add a final markdown cell to your notebook to answer the last two questions.

1.  **Q18: Comparison Table:**
    * Create a markdown table comparing all three models using the results from Part 1 and Part 2.

| Model | Model Type | Best Validation Accuracy | Final Test Accuracy |
| :--- | :--- | :--- | :--- |
| **DeeperResidualCNN** | Custom (Part 1) | [~70% from Part 1] | [Your result from Part 1] |
| **ResNet-34** | Fine-Tuned (Part 2) | [Result from Q16] | [Result from Q17] |
| **EfficientNet-B0**| Fine-Tuned (Part 2) | [Result from Q16] | [Result from Q17] |

2.  **Q19: Discussion**
    * [cite_start]Write a short discussion on the trade-offs, advantages, and limitations, as required [cite: 65-66].
    * **Custom Model (DeeperResidualCNN):**
        * **Advantage:** Full control over architecture.
        * **Limitation:** Reached an accuracy ceiling (~70%). Required significant training (20 epochs) and a deep, complex architecture to even get that far. [cite_start]This aligns with the assignment's note on the difficulty of training from scratch[cite: 56].
    * **Pre-trained Models (ResNet/EfficientNet):**
        * **Advantage:** Achieved *much* higher accuracy (likely 90%+) in *far fewer* epochs (10 vs 20). [cite_start]This is because they were "pretrain[ed]... on an extensive dataset" [cite: 56] and already know how to see basic features.
        * **Limitation:** The models are "black boxes" and can be very large.
        * **Trade-off:** We traded a small amount of model flexibility for a massive gain in performance and a huge reduction in training time. [cite_start]This clearly shows transfer learning is the superior approach when a suitable pre-trained model exists[cite: 56].