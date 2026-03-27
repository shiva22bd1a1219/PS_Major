import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Updated to use relative path to avoid machine-specific errors
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(BASE_DIR, "archive/Training")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory {train_dir} not found. Please restore the archive folder to train.")
        return

    dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    # 200 per class for lightning fast training
    indices = []
    for c in range(4):
        c_idx = [i for i, (_, label) in enumerate(dataset.samples) if label == c][:200]
        indices.extend(c_idx)
    
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(4): # 4 epochs is enough for Transfer Learning
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/4 - Loss: {running_loss/len(dataloader):.4f} - Acc: {100 * correct / total:.2f}%")
        
    save_path = os.path.join(BASE_DIR, "resnet_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"ResNet model saved to {save_path}!")

if __name__ == "__main__":
    train_resnet()
