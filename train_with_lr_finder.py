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
from LR_finder import LRFinder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_optimal_lr(model, train_loader, criterion, device):
    # Initialize optimizer with a very low learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    
    # Initialize the learning rate finder
    lr_finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        memory_cache=True
    )
    
    # Run the learning rate finder
    lr_finder.range_test(
        train_loader,
        end_lr=10,
        num_iter=100,
        step_mode='exp',
        smooth_f=0.05,
        diverge_th=5
    )
    
    # Plot the results
    lr_finder.plot(skip_start=10, skip_end=5)
    plt.savefig('lr_find_results.png')
    plt.close()
    
    # Get the learning rate with minimum loss
    best_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
    
    # Reset the model and optimizer to their initial states
    lr_finder.reset()
    model.train()
    return best_lr

def main():
    cfg = Config()
    device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    
    # Prepare datasett
    transform = get_transforms()
    dataset = InpaintingDataset(cfg['inpainted_dir'], cfg['masks_dir'], cfg['resize_height'], cfg['resize_width'], transform=transform)
    
    total_images = len(dataset)
    val_split = int(total_images * cfg['val_split'])
    train_indices = list(range(0, total_images - val_split))
    val_indices = list(range(total_images - val_split, total_images))
    
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_data,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )
    
    # Model
    model = get_model(cfg)
    model = model.to(device)
    
    # Loss
    if cfg['loss'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg['loss'] == 'dice':
        criterion = DiceLoss()
    elif cfg['loss'] == 'bce_dice':
        criterion = BCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg['loss']}")
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    best_lr = find_optimal_lr(model, train_loader, criterion, device)
    print(f"Optimal learning rate found: {best_lr:.2e}")
    
    # Initialize optimizer and scheduler with found learning rate
    optimizer, scheduler = initialize_optimizer_with_lr(model, best_lr, cfg)

def initialize_optimizer_with_lr(model, lr, cfg):
    """Initialize optimizer and scheduler with specified learning rate"""
    if cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    scheduler = None
    if 'lr_scheduler' in cfg:
        if cfg['lr_scheduler'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.get('step_size', 10),
                gamma=cfg.get('gamma', 0.1)
            )
        elif cfg['lr_scheduler'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=cfg.get('factor', 0.5),
                patience=cfg.get('patience', 3),
                verbose=True
            )
        elif cfg['lr_scheduler'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.get('T_max', 10),
                eta_min=cfg.get('eta_min', 0)
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {cfg['lr_scheduler']}")

    return optimizer, scheduler
    
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
        
        # Step scheduler if present
        if scheduler:
            if cfg['lr_scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()