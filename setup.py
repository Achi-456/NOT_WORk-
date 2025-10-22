import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import torch

# Define Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 9  # RealWaste dataset has 9 classes
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RAW_DATA_DIR = 'DataSet/RealWaste'
SPLIT_DATA_DIR = 'data'

def split_data():
    """
    Split the RealWaste dataset into train (70%), validation (15%), and test (15%) sets
    while maintaining class distribution.
    """
    # Create output directories
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(SPLIT_DATA_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
    
    # Get all class folders
    class_folders = [f for f in os.listdir(RAW_DATA_DIR) 
                    if os.path.isdir(os.path.join(RAW_DATA_DIR, f))]
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    for class_name in class_folders:
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        
        # Get all images in this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        print(f"Class '{class_name}': {len(image_files)} images")
        
        if len(image_files) < 3:
            print(f"Warning: Class '{class_name}' has too few images for splitting")
            continue
        
        # First split: 70% train, 30% temp (which will be split into 15% val, 15% test)
        train_files, temp_files = train_test_split(
            image_files, 
            test_size=0.3, 
            random_state=42,
            stratify=None  # Can't stratify single class
        )
        
        # Second split: 15% validation, 15% test (from the 30% temp)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.5,  # 0.5 of 30% = 15%
            random_state=42
        )
        
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Create class directories in each split
        for split in ['train', 'validation', 'test']:
            class_split_dir = os.path.join(SPLIT_DATA_DIR, split, class_name)
            os.makedirs(class_split_dir, exist_ok=True)
        
        # Copy files to appropriate directories
        for file_path in train_files:
            dst = os.path.join(SPLIT_DATA_DIR, 'train', class_name, os.path.basename(file_path))
            shutil.copy2(file_path, dst)
        
        for file_path in val_files:
            dst = os.path.join(SPLIT_DATA_DIR, 'validation', class_name, os.path.basename(file_path))
            shutil.copy2(file_path, dst)
        
        for file_path in test_files:
            dst = os.path.join(SPLIT_DATA_DIR, 'test', class_name, os.path.basename(file_path))
            shutil.copy2(file_path, dst)
    
    print("Data splitting completed!")
    
    # Print summary
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(SPLIT_DATA_DIR, split)
        total_images = sum([len(os.listdir(os.path.join(split_dir, class_name))) 
                           for class_name in os.listdir(split_dir)])
        print(f"{split.capitalize()} set: {total_images} images")

if __name__ == "__main__":
    print("Setting up project environment...")
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found!")
        exit(1)
    
    print("Starting data split...")
    split_data()
