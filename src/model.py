import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ThermalDefectModel(nn.Module):
    def __init__(self, num_labels: int = 8):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))