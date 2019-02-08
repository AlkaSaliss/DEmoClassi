import sys
sys.path.append('../../vision_utils')
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from custom_torch_utils import initialize_model


class ConvModel(nn.Module):
    """custom Pytorch neural network module"""

    def __init__(self, model_name='resnet', feature_extract=True, num_classes=7, use_pretrained=True):
        super(ConvModel, self).__init__()
        self.model, input_size = initialize_model(model_name, feature_extract, num_classes, 'fer2013', use_pretrained)
        self.input_layer = nn.Conv2d(3, 3, 3, 1, padding=1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.model(x)
        return x


my_model = ConvModel()

# Define the optimizer
optimizer = optim.Adam(
    [
        {"params": my_model.input_layer.parameters(), "lr": 1e-3},
        {"params": my_model.model.fc.parameters(), "lr": 1e-3},
        {"params": my_model.model.conv1.parameters()},
        {"params": my_model.model.layer1.parameters()},
        {"params": my_model.model.layer2.parameters()},
        {"params": my_model.model.layer3.parameters()},
        {"params": my_model.model.layer4.parameters()},
    ],
    lr=1e-6,
)
