import torch
import torch.nn as nn

class MacroNetwork(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        layers = []
        in_channels = 3
        for cfg in layer_config:  # cfg is a list like ['conv3x3', 'maxpool', 'conv5x5']
            if cfg == 'conv3x3':
                layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
            elif cfg == 'conv5x5':
                layers.append(nn.Conv2d(in_channels, 64, kernel_size=5, padding=2))
            elif cfg == 'maxpool':
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.ReLU())
            in_channels = 64
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # global average pooling
        return self.classifier(x)
# Example candidate architecture
arch_config = ['conv3x3', 'conv3x3', 'maxpool', 'conv5x5', 'conv3x3']
model = MacroNetwork(arch_config)
print(f"\nmodel: {model}")

forward = model.forward(torch.randn(1, 3, 32, 32))  # Test forward pass with dummy input
print(f"forward.shape: {forward.shape}")
