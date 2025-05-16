import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_preparation import create_datasets, MedicalDataset
from model import MedicalCNN
import torch.nn as nn

def train_model():
    # Initialize parameters
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DATA_PATH = 'data/train'
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    (train_paths, train_labels), (val_paths, val_labels), (train_trans, val_trans) = create_datasets(DATA_PATH)
    
    # Create dataloaders
    train_dataset = MedicalDataset(train_paths, train_labels, train_trans)
    val_dataset = MedicalDataset(val_paths, val_labels, val_trans)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = MedicalCNN(num_classes=2).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), '../model.pth')
    print("Training complete! Model saved as model.pth")

if __name__ == "__main__":
    train_model()
