#!/usr/bin/env python
import os
import torch
import argparse
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

class EdgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for training the UNet edge detector
        
        Args:
            root_dir (str): Directory with images
            transform: Image transformations
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform
        
        # Create mask directory if it doesn't exist
        os.makedirs(self.mask_dir, exist_ok=True)
        
        # Check if we need to generate masks
        if len(os.listdir(self.mask_dir)) < len(self.images):
            print("Generating edge masks from images using Canny edge detection...")
            self._generate_masks()
    
    def _generate_masks(self):
        """Generate edge masks using Canny edge detection"""
        for img_name in tqdm(self.images, desc="Generating edge masks"):
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
            
            if os.path.exists(mask_path):
                continue
                
            # Load image and generate Canny edge mask
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Save the edge mask
            cv2.imwrite(mask_path, edges)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        
        # If mask doesn't exist, create it
        if not os.path.exists(mask_path):
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            cv2.imwrite(mask_path, edges)
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)
        
        return image, mask

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

def train_model(args):
    """
    Train a UNet model for edge detection
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    transform = {
        "image": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        "mask": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }
    
    # Create dataset
    dataset = EdgeDataset(root_dir=args.data_dir, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = SimpleUNet().to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, masks in progress_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for imgs, masks in progress_bar:
                imgs, masks = imgs.to(device), masks.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * imgs.size(0)
                progress_bar.set_postfix({'loss': loss.item()})
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'unet_edge_detector_best.pth'))
            print(f"  New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'unet_edge_detector.pth'))
    print(f"Final model saved to {os.path.join(args.output_dir, 'unet_edge_detector.pth')}")
    
    # Plot training curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(args.output_dir, 'unet_training_curve.png'))
    plt.show()
    
    return os.path.join(args.output_dir, 'unet_edge_detector.pth')

def main():
    parser = argparse.ArgumentParser(description='Train UNet Edge Detector')
    
    parser.add_argument('--data_dir', type=str, default='data/edge_dataset',
                        help='Directory containing images and masks')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_model(args)
    print(f"Training completed. Model saved to {model_path}")

if __name__ == '__main__':
    main()
