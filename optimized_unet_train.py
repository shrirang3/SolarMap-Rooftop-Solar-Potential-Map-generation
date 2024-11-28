# File: train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from optimized_unet_model import OptimizedUNet  # Changed import
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Added for better training
import warnings
from PIL import Image

# Suppress DeprecationWarnings from Pillow
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Construct label filename
        label_name = img_name.rsplit('.', 1)[0] + '_label.png'
        label_path = os.path.join(self.labels_dir, label_name)
        
        if not os.path.exists(label_path):
            print(f"Label not found for image: {img_name}")
            label = Image.new('L', (512, 512), 0)
        else:
            label = Image.open(label_path).convert('L')

        image = Image.open(img_path).convert('RGB')

        image = image.resize((512, 512), Image.BILINEAR)
        label = label.resize((512, 512), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            label = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)(transforms.ToTensor()(label))

        return image, label

def iou_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    return (intersection + smooth) / (union + smooth)

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - ((2. * intersection + smooth) / (union + smooth))

def train_model(train_dir, val_dir, num_epochs=50, batch_size=8, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = CustomDataset(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model with optimized UNet
    model = OptimizedUNet(n_channels=3, n_classes=1).cuda()
    
    # Combined loss function
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_iou = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            
            # Combined loss
            bce_loss = bce_criterion(outputs, labels)
            dice = dice_loss(outputs, labels)
            loss = 0.5 * bce_loss + 0.5 * dice
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            train_iou += iou_score(outputs, labels)

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                
                bce_loss = bce_criterion(outputs, labels)
                dice = dice_loss(outputs, labels)
                loss = 0.5 * bce_loss + 0.5 * dice
                
                val_loss += loss.item()
                val_iou += iou_score(outputs, labels)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, 'best_unet_model.pth')

    print(f'Best IoU: {best_iou:.4f}')

def test_model(test_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = CustomDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = OptimizedUNet(n_channels=3, n_classes=1).cuda()
    checkpoint = torch.load('best_unet_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0
    test_iou = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            
            bce_loss = nn.BCEWithLogitsLoss()(outputs, labels)
            dice = dice_loss(outputs, labels)
            loss = 0.5 * bce_loss + 0.5 * dice
            
            test_loss += loss.item()
            test_iou += iou_score(outputs, labels)

            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    test_loss /= len(test_loader)
    test_iou /= len(test_loader)

    print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}')

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
    average_precision = average_precision_score(all_labels, all_outputs)

    # Plot precision-recall curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
    plt.savefig('new_precision_recall_curve.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test UNet model')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    train_model(args.train_dir, args.val_dir, args.epochs, args.batch_size, args.lr)
    test_model(args.test_dir, args.batch_size)
