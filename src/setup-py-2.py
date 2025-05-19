import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import shutil
import urllib.request
import zipfile
from torchvision import transforms
import matplotlib.pyplot as plt

def create_project_folders(base_path="./"):
    """Create the project folder structure"""
    folders = [
        "data/yolo_dataset/images/train",
        "data/yolo_dataset/images/val",
        "data/yolo_dataset/labels/train",
        "data/yolo_dataset/labels/val",
        "data/edge_dataset/images",
        "data/edge_dataset/masks",
        "data/rust_level_dataset/train/SA2.5",
        "data/rust_level_dataset/train/SA3.0",
        "data/rust_level_dataset/val/SA2.5",
        "data/rust_level_dataset/val/SA3.0",
        "models",
        "results",
        "sample_data",
        "src"
    ]

    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        
        # Create .gitkeep files to preserve directory structure in git
        gitkeep_path = os.path.join(path, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            pass
    
    print(f"✅ All project folders created successfully inside: {base_path}")
    
    # Create YOLO dataset config file
    yaml_content = """
train: data/yolo_dataset/images/train
val: data/yolo_dataset/images/val
nc: 4
names: ['steel_rust', 'steel_clean', 'nonsteel_rust', 'nonsteel_clean']
"""

    with open(os.path.join(base_path, 'data/yolo_dataset/data.yaml'), 'w') as f:
        f.write(yaml_content)

    print("✅ YOLO dataset config file created successfully")

def download_models(models_dir="models", force_download=False):
    """
    Download pretrained models for the pipeline
    
    Note: This is a placeholder function that would normally download 
    actual pretrained models. For this demo, it creates dummy model files.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Create dummy model files if they don't exist
    model_files = {
        'yolo_best.pt': "YOLO model for steel/rust detection",
        'unet_edge_detector.pth': "UNet model for edge detection",
        'rust_level_classifier.pth': "MobileNetV3 model for rust level classification"
    }
    
    for filename, description in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath) or force_download:
            # In a real application, you would download actual model files here
            print(f"Creating placeholder for {description}...")
            
            # For demo purposes, create dummy model files
            if filename.endswith('.pt'):
                # For YOLO models
                dummy_model = torch.nn.Conv2d(3, 3, 1)
                torch.save(dummy_model.state_dict(), filepath)
            elif filename.endswith('.pth'):
                if 'unet' in filename:
                    # For UNet model
                    class SimpleUNet(torch.nn.Module):
                        def __init__(self):
                            super(SimpleUNet, self).__init__()
                            self.encoder = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.MaxPool2d(2),
                                torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.MaxPool2d(2)
                            )
                            self.decoder = torch.nn.Sequential(
                                torch.nn.ConvTranspose2d(64, 32, 2, stride=2), torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(32, 1, 2, stride=2), torch.nn.Sigmoid()
                            )

                        def forward(self, x):
                            x = self.encoder(x)
                            return self.decoder(x)
                    
                    dummy_model = SimpleUNet()
                    torch.save(dummy_model.state_dict(), filepath)
                else:
                    # For classifier model
                    dummy_model = torch.nn.Linear(1024, 2)
                    torch.save(dummy_model.state_dict(), filepath)
            
            print(f"✅ Created placeholder for {filename}")
        else:
            print(f"✅ Model file {filename} already exists")
    
    print("\n✅ All required model files are available in the models directory")
    print("Note: These are placeholder files. In a real deployment, you would need to train actual models.")

def download_sample_data(base_path="./", force_download=False):
    """
    Download sample data for testing the pipeline
    
    Note: This is a placeholder function that would normally download
    actual sample data. For this demo, it creates dummy sample images.
    """
    sample_dir = os.path.join(base_path, "sample_data")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a few dummy sample images
    for i in range(1, 4):
        img_path = os.path.join(sample_dir, f"sample_{i}.jpg")
        if not os.path.exists(img_path) or force_download:
            # Create a dummy image with random noise
            img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
            
            # Add some shapes to make it more interesting
            if i == 1:
                # Steel with rust
                cv2.rectangle(img, (100, 100), (300, 300), (50, 50, 200), -1)
                cv2.circle(img, (200, 200), 50, (60, 80, 120), -1)
            elif i == 2:
                # Clean steel
                cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
            else:
                # Non-steel
                cv2.circle(img, (208, 208), 150, (100, 180, 100), -1)
            
            cv2.imwrite(img_path, img)
            print(f"✅ Created sample image: {img_path}")
        else:
            print(f"✅ Sample image already exists: {img_path}")
    
    print("\n✅ Sample data created successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup Steel Surface Inspection Robot project')
    parser.add_argument('--base_path', type=str, default='./', help='Base path for project structure')
    parser.add_argument('--force_download', action='store_true', help='Force download/creation of models and sample data')
    
    args = parser.parse_args()
    
    print("Starting setup for Steel Surface Inspection Robot project...")
    
    # Create project folders
    create_project_folders(args.base_path)
    
    # Download/create placeholder models
    download_models(os.path.join(args.base_path, 'models'), args.force_download)
    
    # Download/create sample data
    download_sample_data(args.base_path, args.force_download)
    
    print("\n✅ Setup completed successfully!")
    print("\nTo run the inference pipeline, use:")
    print("python src/inference.py --image sample_data/sample_1.jpg")
    
    print("\nTo train the models, use:")
    print("1. YOLO model: python src/train_yolo.py")
    print("2. UNet edge detector: python src/train_unet.py")
    print("3. Rust classifier: python src/train_classifier.py")
