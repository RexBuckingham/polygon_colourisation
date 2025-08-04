import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import random

COLOR_LIST = ["red", "green", "blue", "yellow", "orange", "magenta", "cyan", "purple"]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_LIST)}

def color_name_to_onehot(color_name):
    idx = COLOR_TO_IDX[color_name]
    one_hot = torch.zeros(len(COLOR_LIST), dtype=torch.float32)
    one_hot[idx] = 1.0
    return one_hot

class PolygonColorDataset(Dataset):
    def __init__(self, input_dir, output_dir, json_path, augment=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augment = augment
        self.resize = transforms.Resize((144, 144))  # Upscale first
        self.to_tensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_img = Image.open(os.path.join(self.input_dir, item["input_polygon"])).convert("RGB")
        output_img = Image.open(os.path.join(self.output_dir, item["output_image"])).convert("RGB")

        input_img = self.resize(input_img)
        output_img = self.resize(output_img)

        if self.augment:
            angle = random.uniform(-30, 30)
            input_img = TF.rotate(input_img, angle)
            output_img = TF.rotate(output_img, angle)

            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                output_img = TF.hflip(output_img)

            if random.random() > 0.5:
                input_img = TF.vflip(input_img)
                output_img = TF.vflip(output_img)

        input_img = self.center_crop(input_img)
        output_img = self.center_crop(output_img)

        input_tensor = self.to_tensor(input_img)
        output_tensor = self.to_tensor(output_img)

        color_onehot = color_name_to_onehot(item["colour"])

        return input_tensor, color_onehot, output_tensor