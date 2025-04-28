import os
import torch
from dataset import InpaintingDataset, get_transforms
from model import get_model
from config import Config
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_mask(mask_tensor, save_path):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    im = Image.fromarray(mask)
    im.save(save_path)

def main():
    cfg = Config()
    device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['inference_output_dir'], exist_ok=True)
    # Prepare dataset
    transform = get_transforms()
    dataset = InpaintingDataset(cfg['inpainted_dir'], cfg['masks_dir'], cfg['resize_height'], cfg['resize_width'], transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Model
    model = get_model(cfg)
    model_path = cfg.get('inference_model_path', None)
    if model_path is None or not os.path.exists(model_path):
        raise ValueError('Set "inference_model_path" in config.yaml to a valid .pth checkpoint')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (image, _) in enumerate(loader):
            image = image.to(device)
            output = model(image)
            output = torch.sigmoid(output)
            save_path = os.path.join(cfg['inference_output_dir'], f"mask_{idx}.png")
            save_mask(output, save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()