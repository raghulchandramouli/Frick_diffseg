# Frick_diffseg

## Overview
this is a modular deep learning pipeline for image inpainting and segmentation. It provides scripts for training and inference using PyTorch, with configurable data augmentation and flexible model selection.

## Project Structure
- `dataset.py`: Dataset class and augmentation pipeline for inpainting and segmentation tasks.
- `train.py`: Script to train the model with configurable parameters.
- `inference.py`: Script to run inference and generate segmentation masks.
- `model.py`: Model architecture and utilities (not shown here, but required).
- `config.py`: Loads configuration from `config.yaml`.
- `config.yaml`: YAML file for all configurable parameters (paths, hyperparameters, etc.).

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**:
   - Python 3.11+
   - PyTorch
   - albumentations
   - tqdm
   - Pillow
   - numpy
   - (Other dependencies as required by your `model.py`)
   
   Install with pip:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your data**:
   - Place inpainted images in the directory specified by `inpainted_dir` in `config.yaml`.
   - Place corresponding masks in the directory specified by `masks_dir`.

## Configuration
All settings are managed via `config.yaml`. Example keys:
- `inpainted_dir`: Path to inpainted images
- `masks_dir`: Path to mask images
- `resize_height`, `resize_width`: Image resize dimensions
- `batch_size`, `epochs`, `learning_rate`, etc.
- `use_cuda`: Whether to use GPU
- `val_split`: Fraction of data for validation
- `loss`, `optimizer`: Loss and optimizer types
- `inference_model_path`: Path to the trained model checkpoint for inference
- `inference_output_dir`: Where to save inference results

## Training
Run:
```bash
python train.py
```
- Model checkpoints will be saved after each epoch.
- Training and validation losses are printed per epoch.

## Inference
After training, set `inference_model_path` in `config.yaml` to your desired checkpoint (e.g., `model_epoch_10.pth`).
Run:
```bash
python inference.py
```
- Output masks will be saved in the directory specified by `inference_output_dir`.

## Customization
- **Augmentations**: Modify `get_transforms()` in `dataset.py` for different data augmentations.
- **Model**: Change or extend `get_model()` in `model.py` for different architectures.
- **Loss/Optimizer**: Add options in `train.py` and `config.yaml`.

## Notes
- Ensure your `config.yaml` is correctly set up before running scripts.
- For best results, match image and mask filenames as expected by `InpaintingDataset`.
- This project assumes masks are single-channel PNGs with values 0 or 255.

## License
This is under MIT License.