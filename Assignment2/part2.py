import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
import numpy as np
import h5py

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
        out = func.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return func.relu(out)

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
            nn.ReLU())
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


class HDF5PCAMDataset(Dataset):
    def __init__(self, h5_file_x, h5_file_y, transform=None):
        self.x_h5 = h5py.File(h5_file_x, 'r')
        self.y_h5 = h5py.File(h5_file_y, 'r')
        self.x_data = self.x_h5['x']
        self.y_data = self.y_h5['y']
        self.transform = transform
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].astype(np.float32) / 255.0
        y = torch.tensor(self.y_data[idx], dtype=torch.long).squeeze()
        x = torch.tensor(x).permute(2, 0, 1)
        if self.transform:
            x = self.transform(x)
        return x, y

def prepare_data_loaders():
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    train_dataset = HDF5PCAMDataset('/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_train_x.h5',
                                '/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_train_y.h5',transform)
    valid_dataset = HDF5PCAMDataset('/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_valid_x.h5',
                                '/kaggle/input/pcam-dataset/camelyonpatch_level_2_split_valid_y.h5')
    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, valid_loader


def train_custom_model(model, train_loader, valid_loader, num_epochs, lr=1e-5):
    criterion = torch.nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = torch.cuda.amp.GradScaler()
    
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}....")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs =  inputs.to(device)
            labels = labels.to(device).float()  # Ensure labels are float
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            correct += ((outputs > 0).float() == labels.unsqueeze(1)).sum().item()  # Threshold at 0
            total += labels.size(0)
        
        train_accuracies.append(100 * correct / total)
        train_losses.append(running_loss / len(train_loader))
        valid_loss, correct, total = 0.0, 0, 0
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))

                valid_loss += loss.item()
                correct += ((outputs > 0).float() == labels.unsqueeze(1)).sum().item()
                total += labels.size(0)
        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(100 * correct / total)
        
        print(f"===Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Accuracy={train_accuracies[-1]:.2f}%, "
              f"Valid Loss={valid_losses[-1]:.4f}, Valid Accuracy={valid_accuracies[-1]:.2f}%===")
    
    
    print(f'\n=====Custom_CLR Model Trained=====\n')

    print(f'Epoch 10: Train Loss = {train_losses[9]:.4f}, Train Accuracy = {train_accuracies[9]:.2f}%,Valid Loss = {valid_losses[9]:.4f}, Valid Accuracy = {valid_accuracies[9]:.2f} ')
    print(f'Epoch 20: Train Loss = {train_losses[19]:.4f}, Train Accuracy = {train_accuracies[19]:.2f}%,Valid Loss = {valid_losses[19]:.4f}, Valid Accuracy = {valid_accuracies[19]:.2f} ')
    print(f'Epoch 25: Train Loss = {train_losses[24]:.4f}, Train Accuracy = {train_accuracies[24]:.2f}%,Valid Loss = {valid_losses[24]:.4f}, Valid Accuracy = {valid_accuracies[24]:.2f} ')

    return model, train_losses, valid_losses, train_accuracies, valid_accuracies

def evaluate_custom_model(model, valid_loader):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            predictions = (outputs > 0).float()  # Threshold at 0
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print("\nTest Results:")
    print(f"Accuracy  = {accuracy * 100:.2f}%")
    print(f"Precision = {precision * 100:.2f}%")
    print(f"Recall    = {recall * 100:.2f}%")
    print(f"F1_Score  = {f1 * 100:.2f}%")
    
def load_custom_model(model_path):
    model = CustomCNNArchitecture()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

if __name__ == "__main__":
    train_loader, valid_loader = prepare_data_loaders()
    model = CustomCNNArchitecture().to(device)

    epochs = 25
    model, train_losses, valid_losses, train_accuracies, valid_accuracies = train_custom_model(model, train_loader, valid_loader,epochs,1e-5)
    torch.save(model.state_dict(), "Custom_CLR_E5_SGD.pth")
    
    print("=====Testing the trained model====")
    evaluate_custom_model(model, valid_loader)

    loaded_model= load_custom_model('/kaggle/working/Custom_CLR_E5_SGD.pth')
    evaluate_custom_model(loaded_model, valid_loader)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(valid_losses, label="Valid Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy", color='green')
    plt.plot(valid_accuracies, label="Valid Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.savefig('Custom_CLR_E5_SGD.jpg', dpi=300, bbox_inches='tight')
    plt.show()