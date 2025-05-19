#!/usr/bin/env python
import os
import argparse
from setup import create_project_folders, download_models, download_sample_data

def main():
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

if __name__ == "__main__":
    main()
