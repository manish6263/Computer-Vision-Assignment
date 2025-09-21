# import dependies

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Pcam Dataset

# Define dataset paths
train_x_path = "/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_train_x.h5"
train_y_path = "/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_train_y.h5"
valid_x_path = "/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_valid_x.h5"
valid_y_path = "/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_valid_y.h5"

# Load data
def load_h5_data(x_path, y_path):
    with h5py.File(x_path, 'r') as f:
        x_data = f["x"][:]  # Convert HDF5 dataset to numpy array
    with h5py.File(y_path, 'r') as f:
        y_data = f["y"][:]  # Labels
    return x_data, y_data

train_x, train_y = load_h5_data(train_x_path, train_y_path)
valid_x, valid_y = load_h5_data(valid_x_path, valid_y_path)

print(f"Train X shape: {train_x.shape}, Train Y shape: {train_y.shape}")
print(f"Valid X shape: {valid_x.shape}, Valid Y shape: {valid_y.shape}")


# Create a Custom PyTorch Dataset

class PCamDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # Get image
        img = Image.fromarray(img)  # Convert to PIL Image

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Binary label
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create dataset instances
train_dataset = PCamDataset(train_x, train_y, transform=transform)
valid_dataset = PCamDataset(valid_x, valid_y, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}")


def get_model(model_name):
    if "ResNet" in model_name:
        model = getattr(models, model_name.lower())(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    elif "VGG" in model_name:
        model = getattr(models, model_name.lower())(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model.to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    
def get_loss_optimizer(model, loss_type="CrossEntropy", optimizer_type="Adam", lr=1e-4):
    if loss_type == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_type == "FocalLoss":
        loss_fn = FocalLoss()

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    return loss_fn, optimizer


import os
import pandas as pd

# Directory to save models and results
save_dir = "/kaggle/working/trained_models"
os.makedirs(save_dir, exist_ok=True)

def train_model(model, train_loader, valid_loader, loss_fn, optimizer, num_epochs=25, lr_scheduler=None, model_name="model"):
    train_acc, valid_acc, train_loss, valid_loss = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}....")
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()  # Convert to LongTensor
            
            labels = labels.squeeze()  # Ensure shape (batch_size,)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)

        # Validation
        model.eval()
        correct, total, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device).long()
                labels = labels.squeeze()  # Ensure shape (batch_size,)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_loss.append(running_loss / len(valid_loader))
        valid_acc.append(100 * correct / total)

        if lr_scheduler:
            lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Accuracy: {train_acc[-1]:.2f}%, Val Accuracy: {valid_acc[-1]:.2f}%")

    # Save model
    model_path = f"/kaggle/working/trained_models/{model_name}_E4_Adam.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return train_acc, valid_acc, train_loss, valid_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_names = ["ResNet18", "ResNet34", "ResNet50", "VGG11", "VGG13", "VGG16", "VGG19"]
# for model_name in model_names:
model_name = "ResNet34"
print(f"\nTraining {model_name}...\n")

model = get_model(model_name)
print('after get model')

loss_fn, optimizer = get_loss_optimizer(model, loss_type="CrossEntropy", optimizer_type="Adam", lr=1e-4)
print('after get loss optimizer')

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print('after StepLR')

train_acc, valid_acc, train_loss, valid_loss = train_model(model, train_loader, valid_loader, loss_fn, optimizer, num_epochs=25, lr_scheduler=lr_scheduler, model_name=model_name)
print('=========Model Trained======')



def evaluate_model(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Convert logits to class index

            y_true.extend(labels.cpu().numpy())  
            y_pred.extend(preds.cpu().numpy())  

    # Convert lists to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Fix shape mismatch by squeezing `y_true`
    y_true = np.squeeze(y_true)  # Removes extra dimensions

    # Debugging step: Check new shapes
    print(f"Fixed y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    # Ensure labels are integers
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    return acc, prec, rec, f1

# Evaluate model
acc, prec, rec, f1 = evaluate_model(model, valid_loader)
# print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-score: {f1:.2f}")
print(f"Accuracy  = {acc * 100:.2f}%")
print(f"Precision = {prec * 100:.2f}%")
print(f"Recall    = {rec * 100:.2f}%")
print(f"F1_Score  = {f1 * 100:.2f}%")


def plot_results(train_acc, valid_acc, train_loss, valid_loss, model_name):
    epochs = range(1, len(train_acc) + 1)
    
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, valid_acc, label="Validation Accuracy", marker='s', linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Epochs for {model_name}")
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Train Loss", marker='o', color='red')
    plt.plot(epochs, valid_loss, label="Validation Loss", marker='s', linestyle='dashed', color='purple')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Epochs for {model_name}")
    plt.legend()
    plt.grid(True)

    plt.show()


plot_results(train_acc, valid_acc, train_loss, valid_loss, model_name)
print("Model:", model_name, "\nTrain Accuracy (Epoch 10):", train_acc[9], "\nTrain Accuracy (Epoch 20):", train_acc[19],"\nTrain Accuracy (Epoch 25):", train_acc[24],"\nValid Accuracy (Epoch 10):", valid_acc[9],"\nValid Accuracy (Epoch 20):", valid_acc[19],"\nValid Accuracy (Epoch 25):", valid_acc[24])