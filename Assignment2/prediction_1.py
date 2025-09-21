import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to dynamically load model
def load_model(model_name, model_path):
    model_dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "vgg11": models.vgg11,
        "vgg13": models.vgg13,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19
    }

    model = model_dict[model_name](pretrained=True) 
    num_features = model.fc.in_features if "resnet" in model_name else model.classifier[0].in_features

    # Modify the last layer for binary classification
    if "resnet" in model_name:
        model.fc = nn.Linear(num_features, 2)
    else:  # For VGG models
        model.classifier[-1] = nn.Linear(num_features, 2)

    # Load pretrained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# Predict function
def predict(model, test_folder, output_file="results_1.txt"):
    results = []
    test_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_images.sort()
    print(f"Processing {len(test_images)} images...")

    for image_name in test_images:
        image_path = os.path.join(test_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, dim=1).item()

        # Convert to positive/negative
        label = "positive" if pred == 1 else "negative"
        results.append(f"{image_name} {label}")

    # Write results to file
    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"Predictions saved to {output_file}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam Test Image Prediction")
    parser.add_argument("model_name", type=str, help="Model type (resnet18, resnet34, resnet50, vgg11, vgg13)")
    parser.add_argument("model_path", type=str, help="Path to the trained model (.pth file)")
    parser.add_argument("test_folder", type=str, help="Path to the folder containing test images")
    
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_name, args.model_path)

    # Run prediction
    predict(model, args.test_folder)
