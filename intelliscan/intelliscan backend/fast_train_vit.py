import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import importlib.util

# Load the local transformer.py
spec = importlib.util.spec_from_file_location(
    "transformer", 
    "C:/Users/Shiva/Desktop/mlml/PS!major/intelliscan/intelliscan backend/transformer.py"
)
transformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer)

TumorClassifierViT = transformer.TumorClassifierViT

def train_vit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Very robust data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dir = "C:/Users/Shiva/Desktop/mlml/PS!major/intelliscan/intelliscan backend/archive/Training"
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    # We only use a subset for fast training today (100 per class = 400 total)
    # The user has an urgent deadline
    indices = []
    for c in range(4):
        c_idx = [i for i, (_, label) in enumerate(dataset.samples) if label == c][:150]
        indices.extend(c_idx)
    
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    
    print(f"Fine-tuning ViT on {len(subset)} images...")
    
    model = TumorClassifierViT(num_classes=4).to(device)
    
    # Try to load existing weights if possible
    model_path = "C:/Users/Shiva/Desktop/mlml/PS!major/intelliscan/intelliscan backend/best_model.pth"
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("Loaded previous ViT weights.")
    except Exception as e:
        print(f"Could not load previous weights: {e}. Training from scratch.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(dataloader):
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
            
        print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(dataloader):.4f} - Acc: {100 * correct / total:.2f}%")
        
    torch.save(model.state_dict(), model_path)
    print("Saved fine-tuned best_model.pth!")

if __name__ == "__main__":
    train_vit()
