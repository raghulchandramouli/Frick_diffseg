import os
import torch
from torch.utils.data import DataLoader
from dataset import InpaintingDataset, get_transforms
from model import get_model
from config import Config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from losses import DiceLoss, BCEDiceLoss

def main():
    cfg = Config()
    device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    # Prepare dataset
    transform = get_transforms()
    dataset = InpaintingDataset(cfg['inpainted_dir'], cfg['masks_dir'], cfg['resize_height'], cfg['resize_width'], transform=transform)
    total_images = len(dataset)
    val_split = int(total_images * cfg['val_split'])
    train_indices = list(range(0, total_images - val_split))
    val_indices = list(range(total_images - val_split, total_images))
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
    val_loader = DataLoader(val_data, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
    # Model
    model = get_model(cfg)
    model = model.to(device)
    # Loss and optimizer
    if cfg['loss'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg['loss'] == 'dice':
        criterion = DiceLoss()
    elif cfg['loss'] == 'bce_dice':
        criterion = BCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg['loss']}")
    if cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
    # Training loop
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        # Save checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()