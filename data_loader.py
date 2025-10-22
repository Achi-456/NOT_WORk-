import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from setup import IMAGE_SIZE, BATCH_SIZE, SPLIT_DATA_DIR

def get_data_loaders():
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Returns:
        train_loader: DataLoader for training set with augmentation
        val_loader: DataLoader for validation set 
        test_loader: DataLoader for test set
        class_names: List of class names
    """
    
    # Define transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Define transforms for validation and test (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=f'{SPLIT_DATA_DIR}/train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f'{SPLIT_DATA_DIR}/validation',
        transform=val_test_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=f'{SPLIT_DATA_DIR}/test',
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get class names
    class_names = train_dataset.classes
    
    print(f"Dataset loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Test the data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    
    # Test loading a batch
    train_iter = iter(train_loader)
    images, labels = next(train_iter)
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")
