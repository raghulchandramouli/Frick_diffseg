import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import JaccardIndex
import time
from LR_finder import LRFinder
import torch.optim as optim

def visualize_sample(model, dataset, device, idx=0):
    image, gt_mask = dataset[idx]
    image_orig = image.permute(1, 2, 0).cpu().numpy()  # convert from tensor to numpy image
    image_display = (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())

    model.eval()
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_display)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.squeeze(0).cpu(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(preds, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()

def train_and_validate(model, train_loader, val_loader, num_epochs, device, criterion, scheduler_type=None, early_stopping=None):
    train_iou_metric = JaccardIndex(task="binary", num_classes=2).to(device)
    val_iou_metric = JaccardIndex(task="binary", num_classes=2).to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}

    # Find optimal learning rate
    print("Finding optimal learning rate...")
    init_optimizer = optim.Adam(model.parameters(), lr=1e-7)
    lr_finder = LRFinder(model=model, optimizer=init_optimizer, criterion=criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode='exp')
    best_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
    lr_finder.plot(skip_start=10, skip_end=5)
    plt.savefig('lr_find_results.png')
    plt.close()
    lr_finder.reset()
    print(f"Optimal learning rate found: {best_lr:.2e}")

    # Initialize optimizer with found learning rate
    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    # Setup scheduler
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        scheduler = None

    print(f"Starting training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        train_iou_metric.reset()

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            images = images.to(device)
            masks = masks.long().squeeze(1).to(device)  # assuming masks shape: [B, 1, H, W]

            optimizer.zero_grad()
            outputs = model(images)  # outputs shape: [B, 2, 512, 512]
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_iou_metric.update(preds, masks)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_iou_score = train_iou_metric.compute().item()
        history['train_loss'].append(epoch_loss)
        history['train_iou'].append(train_iou_score)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou_metric.reset()
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                masks_long = masks.long().squeeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks_long)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_iou_metric.update(preds, masks_long)

        val_loss_avg = val_loss / len(val_loader.dataset)
        val_iou_score = val_iou_metric.compute().item()
        history['val_loss'].append(val_loss_avg)
        history['val_iou'].append(val_iou_score)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed:.2f}s - Train Loss: {epoch_loss:.4f} - Train IoU: {train_iou_score:.4f} - Val Loss: {val_loss_avg:.4f} - Val IoU: {val_iou_score:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss_avg)
            else:
                scheduler.step()

        if early_stopping is not None:
            stop, improved = early_stopping.step(val_iou_score, model)
            if improved:
                print(f"Validation IoU improved to {val_iou_score:.4f}")
            if stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("Training Completed!")
    return history, model