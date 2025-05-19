#!/usr/bin/env python
import os
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import yaml

def train_yolo_model(args):
    """
    Train a YOLO model for steel and rust detection
    
    Args:
        args: Command line arguments
    """
    # Create data.yaml file
    if not os.path.exists(os.path.join(args.data_dir, 'data.yaml')):
        yaml_content = f"""
train: {os.path.join(args.data_dir, 'images/train')}
val: {os.path.join(args.data_dir, 'images/val')}
nc: 4
names: ['steel_rust', 'steel_clean', 'nonsteel_rust', 'nonsteel_clean']
"""
        with open(os.path.join(args.data_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
        print(f" Created data.yaml file at {os.path.join(args.data_dir, 'data.yaml')}")
    
    # Check if dataset exists
    train_img_dir = os.path.join(args.data_dir, 'images/train')
    val_img_dir = os.path.join(args.data_dir, 'images/val')
    
    if not os.path.exists(train_img_dir) or not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"Dataset not found at {args.data_dir}. Please ensure the dataset is properly structured.")
    
    train_imgs = [f for f in os.listdir(train_img_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
    val_imgs = [f for f in os.listdir(val_img_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    if len(train_imgs) == 0 or len(val_imgs) == 0:
        raise ValueError(f"No images found in the dataset directories. Please check your dataset.")
    
    print(f"Found {len(train_imgs)} training images and {len(val_imgs)} validation images.")
    
    # Load base model
    model = YOLO(args.base_model)
    print(f" Loaded base model: {args.base_model}")
    
    # Train the model
    results = model.train(
        data=os.path.join(args.data_dir, 'data.yaml'),
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch_size,
        patience=args.patience,
        optimizer=args.optimizer,
        verbose=True,
        project=args.output_dir,
        name='yolo_train',
        exist_ok=True
    )
    
    # Get best model path
    best_model_path = model.ckpt_path
    if not best_model_path or not os.path.exists(best_model_path):
        best_model_path = os.path.join(args.output_dir, 'yolo_train', 'weights', 'best.pt')
    
    # Copy best model to the expected location
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    import shutil
    shutil.copy(best_model_path, os.path.join(args.output_dir, 'yolo_best.pt'))
    
    print(f" Best model saved to {os.path.join(args.output_dir, 'yolo_best.pt')}")
    
    # Export to ONNX if requested
    if args.export_onnx:
        model.export(format='onnx')
        print(" Model exported to ONNX format.")
    
    return os.path.join(args.output_dir, 'yolo_best.pt')

def main():
    parser = argparse.ArgumentParser(description='Train YOLO Steel/Rust Detector')
    
    parser.add_argument('--data_dir', type=str, default='data/yolo_dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--base_model', type=str, default='yolov8n.pt',
                        help='Base model to use for training')
    parser.add_argument('--img_size', type=int, default=416,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='Optimizer to use for training')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format after training')
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_yolo_model(args)
    print(f"Training completed. Model saved to {model_path}")
    
    # Run inference on sample image if available
    sample_dir = 'sample_data'
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
        if samples:
            print("\nRunning inference on sample images:")
            model = YOLO(model_path)
            for sample in samples[:3]:  # Test on up to 3 samples
                sample_path = os.path.join(sample_dir, sample)
                results = model(sample_path)
                
                # Show results
                result_img = results[0].plot()
                cv2.imwrite(f"results/yolo_inference_{sample}", result_img)
                print(f"  Inference result saved to results/yolo_inference_{sample}")

if __name__ == '__main__':
    main()
