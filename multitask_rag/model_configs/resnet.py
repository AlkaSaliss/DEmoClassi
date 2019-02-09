import sys
sys.path.append('../../vision_utils')
import torch.optim as optim
from vision_utils.custom_torch_utils import initialize_model
import torch.nn as nn


class ConvModelMultiTask(nn.Module):
    """custom Pytorch neural network module for multitask learning"""

    def __init__(self, model_name='resnet', feature_extract=True, num_classes=7, use_pretrained=True):
        super(ConvModelMultiTask, self).__init__()
        self.conv_base, input_size = initialize_model(model_name, feature_extract, num_classes, 'utk', use_pretrained)
        self.output_age = nn.Linear(128, 1)
        self.output_gender = nn.Linear(128, 1)
        self.output_race = nn.Linear(128, 4)

    def forward(self, age, gender, race):
        age = self.output_age(self.conv_base(age))
        gender = self.output_gender(self.conv_base(gender))
        race = self.output_race(self.conv_base(race))
        return age, gender, race