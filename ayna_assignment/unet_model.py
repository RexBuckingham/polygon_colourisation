import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""
    def __init__(self, feature_channels, condition_dim):
        super().__init__()
        self.feature_channels = feature_channels
        
        # Generate gamma (scale) and beta (shift) parameters from condition
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, feature_channels * 2),
            nn.ReLU(),
            nn.Linear(feature_channels * 2, feature_channels * 2)
        )
        
    def forward(self, features, condition):
        # Generate FiLM parameters
        film_params = self.film_generator(condition)  # (B, feature_channels * 2)
        
        # Split into gamma and beta
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # Each: (B, feature_channels)
        
        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM: gamma * features + beta
        return gamma * features + beta

class DoubleConvFiLM(nn.Module):
    """DoubleConv block with FiLM conditioning"""
    def __init__(self, in_c, out_c, condition_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.film1 = FiLMLayer(out_c, condition_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.film2 = FiLMLayer(out_c, condition_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, condition):
        # First conv + BN + FiLM + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film1(x, condition)
        x = self.relu1(x)
        
        # Second conv + BN + FiLM + ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.film2(x, condition)
        x = self.relu2(x)
        
        return x
    
class UNetFullFiLM(nn.Module):
    def __init__(self, in_channels=3, cond_dim=8):
        super().__init__()
        self.cond_dim = cond_dim
        
        self.color_embed = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.down1 = DoubleConvFiLM(in_channels, 64, condition_dim=64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConvFiLM(64, 128, condition_dim=64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConvFiLM(128, 256, condition_dim=64)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConvFiLM(256, 512, condition_dim=64)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvFiLM(512, 1024, condition_dim=64)

        self.up3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec3 = DoubleConvFiLM(1024, 512, condition_dim=64)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConvFiLM(512, 256, condition_dim=64)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DoubleConvFiLM(256, 128, condition_dim=64)
        
        self.up0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec0 = DoubleConvFiLM(128, 64, condition_dim=64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x_img, x_color):
        color_embed = self.color_embed(x_color)

        d1 = self.down1(x_img, color_embed)
        d2 = self.down2(self.pool1(d1), color_embed)
        d3 = self.down3(self.pool2(d2), color_embed)
        d4 = self.down4(self.pool3(d3), color_embed)

        bottleneck = self.bottleneck(self.pool4(d4), color_embed)

        u3 = self.up3(bottleneck)
        u3 = torch.cat([u3, d4], dim=1)
        u3 = self.dec3(u3, color_embed)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.dec2(u2, color_embed)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.dec1(u1, color_embed)

        u0 = self.up0(u1)
        u0 = torch.cat([u0, d1], dim=1)
        u0 = self.dec0(u0, color_embed)

        return self.out(u0)