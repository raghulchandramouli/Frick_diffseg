import torch
import torch.nn as nn
import torchvision.models as models

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        super().__init__()
        self.encoder = models.resnet34(pretrained=pretrained)
        self.inc = UNetBlock(n_channels, 64)
        self.down1 = UNetBlock(64, 128)
        self.down2 = UNetBlock(128, 256)
        self.down3 = UNetBlock(256, 512)
        self.down4 = UNetBlock(512, 512)
        self.up1 = UNetBlock(512+512, 256)
        self.up2 = UNetBlock(256+256, 128)
        self.up3 = UNetBlock(128+128, 64)
        self.up4 = UNetBlock(64+64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.maxpool(x1))
        x3 = self.down2(self.maxpool(x2))
        x4 = self.down3(self.maxpool(x3))
        x5 = self.down4(self.maxpool(x4))
        x = self.up(self.up1(torch.cat([x5, x4], 1)))
        x = self.up(self.up2(torch.cat([x, x3], 1)))
        x = self.up(self.up3(torch.cat([x, x2], 1)))
        x = self.up(self.up4(torch.cat([x, x1], 1)))
        logits = self.outc(x)
        return logits

def get_model(cfg):
    model_cfg = cfg['model']
    if model_cfg['name'].lower() == 'unet':
        return UNet(n_channels=3, n_classes=1, pretrained=model_cfg.get('pretrained', True))
    else:
        raise ValueError(f"Unknown model: {model_cfg['name']}")