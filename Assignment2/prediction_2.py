from PIL import Image
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class InceptionModule(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channel, 64, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, 64, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch_pool(x)
        ], dim=1)

class CustomCNNArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2)
        )
        self.inception = InceptionModule(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.inception(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

import torch
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms

def load_model(model_path, device):
    """
    Load the pretrained custom model.
    """
    model = CustomCNNArchitecture()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def transform_image(image_path):
    """
    Apply necessary transformations to the image.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_image(model, image_path, device):
    """
    Make prediction for a single image.
    Returns "Positive" or "Negative".
    """
    image = transform_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        prediction = (torch.sigmoid(outputs) > 0.5).float()  # Threshold at 0.5
        return "Positive" if prediction.item() == 1 else "Negative"

def predict_folder(model, folder_path, output_file, device):
    """
    Predict all images in a folder and save results as positive/negative.
    """
    image_files = sorted([f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    with open(output_file, 'w') as f:
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            prediction = predict_image(model, img_path, device)
            f.write(f"{img_file},{prediction}\n")

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam Test Image Prediction")

    parser.add_argument('test_folder', type=str, help='Path to folder containing test images')
    parser.add_argument('path_to_model', type=str, help='Path to pretrained custom model')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.path_to_model, device)

    output_file = 'results_2.txt'
    predict_folder(model, args.test_folder, output_file, device)
    print(f"Predictions saved to {output_file}")