import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


from dataset_loader import color_name_to_onehot 
from model_architecture import UNetFullFiLM 

transform = T.Compose([T.Resize((144, 144)), T.CenterCrop(128), T.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_film(image_path, color_name, model_path, cond_dim=8):
    """Test your UNetFullFiLM (FiLM-everywhere UNet) with RGB output"""
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    color_vector = color_name_to_onehot(color_name).unsqueeze(0).to(device)

    # Initialize the Full FiLM model
    model = UNetFullFiLM(in_channels=3, cond_dim=cond_dim).to(device)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded FiLM model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Inference
    with torch.no_grad():
        pred = model(img_tensor, color_vector)
        pred_np = pred.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_np = np.clip(pred_np, 0, 1)  # Safe for RGB space

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Polygon", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_np)
    plt.title(f"Predicted: {color_name}", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return pred_np

#prediction used ms paint to draw random shape and checked the predicted output
result = predict_film(
    image_path="/content/drive/MyDrive/ayna_dataset/validation/inputs/random_test_unseen.png",
    color_name="magenta",
    model_path="/content/drive/MyDrive/unet_filmorg.pth",
    cond_dim=8 
)
