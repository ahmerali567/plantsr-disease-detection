import torch
import torch.nn as nn
import torchvision.models as models

# --- Generator Architecture: RRDB-based ESRGAN-lite ---
class RRDBBlock(nn.Module):
    """ Residual in Residual Dense Block for high-frequency detail preservation """
    def __init__(self, channels):
        super(RRDBBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.conv(x)

class TaskAwareGenerator(nn.Module):
    """ Generator that produces 512x512 SR images from 128x128 LR inputs """
    def __init__(self):
        super(TaskAwareGenerator, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[RRDBBlock(64) for _ in range(5)])
        
        # Upsampling layers using PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        return torch.sigmoid(self.upsample(x))

# --- Classifier Architecture: ResNet/MobileNet based Disease Detector ---
class DiseaseClassifier(nn.Module):
    """ Task-aware classifier to identify plant diseases from SR images """
    def __init__(self, num_classes=9):
        super(DiseaseClassifier, self).__init__()
        # Using a lightweight MobileNetV3 for efficient joint training
        self.base_model = models.mobilenet_v3_small(weights='DEFAULT')
        self.base_model.classifier[3] = nn.Linear(self.base_model.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)