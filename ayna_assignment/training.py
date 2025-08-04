import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import wandb
import os

from dataset import PolygonColorDataset, COLOR_LIST, color_name_to_onehot 
from unet_model import UNetFullFiLM 

# Define paths from Google Drive
DATASET_ROOT = "/content/drive/MyDrive/ayna_dataset"
MODEL_SAVE_PATH = "/content/drive/MyDrive/unet_filmorg.pth"

# Initialize wandb
wandb.init(project="ayna-dec-film")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Improved loss function for color segmentation
class ColorSegmentationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for pixel-wise loss
        self.beta = beta    # Weight for color consistency loss
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, color_condition):
        # Primary pixel-wise loss
        pixel_loss = self.mse(pred, target)
        
        # Color consistency loss - penalize color deviations within regions
        # Compute gradient-based edge mask to identify polygon boundaries
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        
        # Simple gradient-based boundary detection
        pred_grad_x = torch.abs(pred_gray[:, :, :, 1:] - pred_gray[:, :, :, :-1])
        pred_grad_y = torch.abs(pred_gray[:, :, 1:, :] - pred_gray[:, :, :-1, :])
        
        # Mask non-boundary regions for color consistency
        interior_mask_x = (pred_grad_x < 0.05).float() 
        interior_mask_y = (pred_grad_y < 0.05).float() 
        
        # Color consistency within regions
        consistency_loss_x = torch.mean(interior_mask_x * pred_grad_x)
        consistency_loss_y = torch.mean(interior_mask_y * pred_grad_y)
        
        consistency_loss = consistency_loss_x + consistency_loss_y
        
        return self.alpha * pixel_loss + self.beta * consistency_loss

def calculate_color_accuracy(pred, target, threshold=0.1):
    """Calculate pixel-wise color accuracy"""
    color_diff = torch.norm(pred - target, dim=1)  # L2 distance in RGB space
    accurate_pixels = (color_diff < threshold).float()
    return torch.mean(accurate_pixels)

def calculate_region_color_accuracy(pred, target):
    """Calculate color accuracy within detected regions"""
    pred_flat = pred.view(-1, 3)
    target_flat = target.view(-1, 3)
    
    color_similarity = 1.0 - torch.norm(pred_flat - target_flat, dim=1) / (3.0 ** 0.5)
    return torch.mean(color_similarity)

# Load datasets
train_dataset = PolygonColorDataset(
    input_dir=os.path.join(DATASET_ROOT, "training/inputs"),
    output_dir=os.path.join(DATASET_ROOT, "training/outputs"),
    json_path=os.path.join(DATASET_ROOT, "training/data.json"),
    augment=True
)
val_dataset = PolygonColorDataset(
    input_dir=os.path.join(DATASET_ROOT, "validation/inputs"),
    output_dir=os.path.join(DATASET_ROOT, "validation/outputs"),
    json_path=os.path.join(DATASET_ROOT, "validation/data.json"),
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Initialize model with FiLM
model = UNetFullFiLM(in_channels=3, cond_dim=len(COLOR_LIST)).to(device) 
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Use improved loss function
criterion = ColorSegmentationLoss(alpha=1.0, beta=2.0)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

# Training loop
best_val_loss = float('inf')

for epoch in range(1, 76):
    model.train()
    total_loss = 0
    total_color_acc = 0
    total_region_acc = 0
    
    for x_img, x_color, y in train_loader:
        x_img, x_color, y = x_img.to(device), x_color.to(device), y.to(device)

        out = model(x_img, x_color)
        loss = criterion(out, y, x_color)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate color-specific metrics
        with torch.no_grad():
            color_acc = calculate_color_accuracy(out, y)
            region_acc = calculate_region_color_accuracy(out, y)
            total_color_acc += color_acc.item()
            total_region_acc += region_acc.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_color_acc = total_color_acc / len(train_loader)
    avg_region_acc = total_region_acc / len(train_loader)
    
    wandb.log({
        "train_loss": avg_train_loss,
        "train_color_accuracy": avg_color_acc,
        "train_region_accuracy": avg_region_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Validation
    model.eval()
    val_loss = 0
    val_color_acc = 0
    val_region_acc = 0
    
    with torch.no_grad():
        for x_img, x_color, y in val_loader:
            x_img, x_color, y = x_img.to(device), x_color.to(device), y.to(device)
            out = model(x_img, x_color)
            val_loss += criterion(out, y, x_color).item()
            
            # Validation metrics
            color_acc = calculate_color_accuracy(out, y)
            region_acc = calculate_region_color_accuracy(out, y)
            val_color_acc += color_acc.item()
            val_region_acc += region_acc.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_color_acc = val_color_acc / len(val_loader)
    avg_val_region_acc = val_region_acc / len(val_loader)
    
    wandb.log({
        "val_loss": avg_val_loss,
        "val_color_accuracy": avg_val_color_acc,
        "val_region_accuracy": avg_val_region_acc
    })
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, MODEL_SAVE_PATH)
        print(f"✓ New best model saved at epoch {epoch}")

    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Color Acc: {avg_val_color_acc:.3f} | Region Acc: {avg_val_region_acc:.3f}")

# Save final model
final_path = MODEL_SAVE_PATH.replace('.pth', '_final.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_loss': avg_val_loss,
}, final_path)

print(f"Training completed!")
print(f"Best model saved to: {MODEL_SAVE_PATH}")
print(f"Final model saved to: {final_path}")