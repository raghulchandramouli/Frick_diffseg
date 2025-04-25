import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class InpaintingDataset(Dataset):
    def __init__(self, images_dir, masks_dir, resize_height, resize_width, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        base = os.path.splitext(image_name)[0]
        if 'inpainted_' in base:
            base = base.replace('inpainted_', '')
        mask_name = f"mask_{base}.png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        image = A.Resize(self.resize_height, self.resize_width)(image=image)["image"]
        mask = A.Resize(self.resize_height, self.resize_width)(image=mask)["image"]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.7),
        A.RandomRotate90(p=0.8),
        A.Affine(translate_percent=0.1, scale=[0.9, 1.1], rotate=(-60, 60), p=0.6),
        A.ElasticTransform(p=0.4, alpha=120, sigma=120 * 0.05),
        A.GridDistortion(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(p=0.4),
        A.Perspective(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], is_check_shapes=False)