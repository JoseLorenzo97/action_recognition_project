import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=512, num_layers=2):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x)
        features = features.view(b, t, -1)

        _, (h_n, _) = self.lstm(features)

        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits
