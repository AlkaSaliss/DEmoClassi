import sys
sys.path.append('../../vision_utils')
import torch.optim as optim
from vision_utils.custom_torch_utils import initialize_model
import torch.nn as nn


class ConvModelMultiTask(nn.Module):
    """custom Pytorch neural network module for multitask learning"""

    def __init__(self, model_name='vgg', feature_extract=True, use_pretrained=True):
        super(ConvModelMultiTask, self).__init__()
        self.conv_base, input_size = initialize_model(model_name, feature_extract, 'utk', use_pretrained)
        self.conv_base
        self.output_age = nn.Linear(128, 1)
        self.output_gender = nn.Linear(128, 2)
        self.output_race = nn.Linear(128, 5)

    def forward(self, age, gender, race):
        age = self.output_age(self.conv_base(age))
        gender = self.output_gender(self.conv_base(gender))
        race = self.output_race(self.conv_base(race))
        return age, gender, race


my_model = ConvModelMultiTask()
# Define the optimizer
optimizer = optim.Adam(
    [
        {"params": my_model.conv_base.classifier[6].parameters(), "lr": 1e-3},
        {"params": my_model.output_age.parameters(), "lr": 1e-3},
        {"params": my_model.output_gender.parameters(), "lr": 1e-3},
        {"params": my_model.output_race.parameters(), "lr": 1e-3},
        {"params": my_model.conv_base.classifier[0].parameters()},
        {"params": my_model.conv_base.classifier[3].parameters()},
        {"params": my_model.conv_base.features.parameters()}
    ],
    lr=1e-6,
)
