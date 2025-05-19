import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import albumentations as A

class RustLevelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        self.class_names = ["SA2.5", "SA3.0"]

        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.data.append((os.path.join(class_dir, img_name), label))
        
        if len(self.data) == 0:
            raise ValueError(f"No images found in {root_dir}")
            
        print(f"Loaded {len(self.data)} images from dataset")
        
        # Print class distribution
        class_counts = {}
        for _, label in self.data:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images ({count/len(self.data)*100:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(args):
    """
    Train a rust level classifier model
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # If val directory doesn't exist, use train-val split
    if not os.path.exists(val_dir):
        print("Validation directory not found. Using 80-20 train-val split from training data.")
        dataset = RustLevelDataset(root_dir=train_dir, transform=train_transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        val_dataset.transform = val_transform
    else:
        train_dataset = RustLevelDataset(root_dir=train_dir, transform=train_transform)
        val_dataset = RustLevelDataset(root_dir=val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    if args.model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(1024, 2)  # 2 classes: SA2.5 and SA3.0
    elif args.model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(1280, 2)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = ''
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': val_correct / val_total
                })
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'rust_level_classifier_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'rust_level_classifier_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Also save as the expected name for the inference pipeline
    standard_model_path = os.path.join(args.output_dir, 'rust_level_classifier.pth')
    torch.save(model.state_dict(), standard_model_path)
    print(f"Standard model saved to {standard_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.show()
    
    return best_model_path
    
def main():
    parser = argparse.ArgumentParser(description='Train Rust Level Classifier')
    
    parser.add_argument('--data_dir', type=str, default='data/rust_level_dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--model_type', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    args = parser.parse_args()
    
    # Train model
    best_model_path = train_model(args)
    print(f"Training completed. Best model saved to {best_model_path}")

if __name__ == '__main__':
    main()
