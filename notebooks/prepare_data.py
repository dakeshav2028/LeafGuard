import os
import shutil
import random
from pathlib import Path
import kagglehub

def prepare_dataset():
    """
    Downloads the emmarex/plantdisease dataset, extracts the Potato subfolders,
    and splits a subset of 200 images per class into train, validation, and test sets.
    """
    print("Step 1: Downloading dataset from Kaggle...")
    # kagglehub.dataset_download downloads the dataset without requiring API credentials
    dataset_path = kagglehub.dataset_download("emmarex/plantdisease")
    print(f"Dataset downloaded successfully to: {dataset_path}")
    
    # We define the classes we want to extract
    classes = [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy"
    ]
    
    # Define target path for processed data
    project_data_dir = Path("c:/MY_PROJECTS/Image_Classification_web_app/data")
    
    # Clean previous data if any
    for split in ["train", "val", "test"]:
        split_dir = project_data_dir / split
        if split_dir.exists():
            print(f"Cleaning existing directory: {split_dir}")
            shutil.rmtree(split_dir)
            
    # Set random seed for reproducibility
    random.seed(42)
    
    # We will limit the images per class to 200 to prevent CPU training from taking too long.
    MAX_IMAGES_PER_CLASS = 200
    
    # Splits configuration: 70% Train, 15% Val, 15% Test
    TRAIN_PCT = 0.70
    VAL_PCT = 0.15
    # TEST_PCT = 0.15 (the remainder)
    
    print("\nStep 2: Processing and splitting images...")
    
    # Locate downloaded folders
    src_root = Path(dataset_path)
    # Sometimes kagglehub returns a directory with nested folders, we look for class subfolders
    # Let's inspect src_root subdirectories
    subdirs = [d for d in src_root.iterdir() if d.is_dir()]
    
    # Handle nested directories if any
    # If the root folder has only one child and it's a directory, go inside
    if len(subdirs) == 1 and subdirs[0].name.lower() in ["plantdisease", "plantvillage"]:
        src_root = subdirs[0]
        subdirs = [d for d in src_root.iterdir() if d.is_dir()]
    
    # Check if there's a sub-directory named plantdisease inside
    plantdisease_dir = src_root / "plantdisease"
    if plantdisease_dir.exists() and plantdisease_dir.is_dir():
        src_root = plantdisease_dir
    
    print(f"Source folder for extraction: {src_root}")
    
    for cls in classes:
        src_class_dir = src_root / cls
        
        # In some versions of this dataset, folder naming might vary slightly, e.g., double underscores or casing.
        # Let's double check matches if direct path doesn't exist
        if not src_class_dir.exists():
            # Try finding folder case-insensitively or with matching start/end
            matched = False
            for d in src_root.iterdir():
                if d.is_dir() and d.name.lower().replace("_", "") == cls.lower().replace("_", ""):
                    src_class_dir = d
                    matched = True
                    break
            if not matched:
                print(f"WARNING: Source folder for class '{cls}' not found in {src_root}!")
                continue
        
        # Get all image files in the source class folder
        image_files = [f for f in src_class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.jpg_']]
        print(f"Found {len(image_files)} total images for class '{cls}' in source.")
        
        # Shuffle images for random selection
        random.shuffle(image_files)
        
        # Subset to MAX_IMAGES_PER_CLASS
        subset_images = image_files[:MAX_IMAGES_PER_CLASS]
        n_images = len(subset_images)
        print(f"Selected {n_images} images for class '{cls}' subset.")
        
        # Calculate split sizes
        n_train = int(n_images * TRAIN_PCT)
        n_val = int(n_images * VAL_PCT)
        
        # Split the list
        train_images = subset_images[:n_train]
        val_images = subset_images[n_train:n_val + n_train]
        test_images = subset_images[n_train + n_val:]
        
        print(f"Splits for {cls}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
        
        # Copy to folders
        for split_name, split_list in [("train", train_images), ("val", val_images), ("test", test_images)]:
            target_class_dir = project_data_dir / split_name / cls
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_list:
                # Copy file and preserve original filename
                shutil.copy2(img_path, target_class_dir / img_path.name)
                
    print("\nDataset preparation completed successfully!")
    print(f"Train path: {project_data_dir / 'train'}")
    print(f"Val path: {project_data_dir / 'val'}")
    print(f"Test path: {project_data_dir / 'test'}")

if __name__ == "__main__":
    prepare_dataset()
