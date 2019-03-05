import sys
sys.path.append('../../vision_utils')
import torch.optim as optim
from vision_utils.custom_torch_utils import initialize_model
import torch.nn as nn


class ConvModelMultiTask(nn.Module):
    """custom Pytorch neural network module for multitask learning"""

    def __init__(self, model_name='resnet', feature_extract=True, use_pretrained=True):
        super(ConvModelMultiTask, self).__init__()
        self.conv_base, input_size = initialize_model(model_name, feature_extract, 'utk', use_pretrained)
        self.output_age = nn.Linear(128, 1)
        self.output_gender = nn.Linear(128, 2)
        self.output_race = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv_base(x)
        age = self.output_age(x)
        gender = self.output_gender(x)
        race = self.output_race(x)
        return age, gender, race


my_model = ConvModelMultiTask()
# Define the optimizer
optimizer = optim.Adam(
    [
        {"params": my_model.conv_base.fc.parameters(), "lr": 1e-3},
        {"params": my_model.output_age.parameters(), "lr": 1e-3},
        {"params": my_model.output_gender.parameters(), "lr": 1e-3},
        {"params": my_model.output_race.parameters(), "lr": 1e-3},
        {"params": my_model.conv_base.conv1.parameters()},
        {"params": my_model.conv_base.layer1.parameters()},
        {"params": my_model.conv_base.layer2.parameters()},
        {"params": my_model.conv_base.layer3.parameters()},
        {"params": my_model.conv_base.layer4.parameters()},
    ],
    lr=1e-6,
)
