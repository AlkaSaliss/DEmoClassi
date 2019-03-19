import torch.optim as optim
from vision_utils.custom_torch_utils import initialize_model


MODEL_NAME = 'resnet'
FEATURE_EXTRACT = True
NUM_CLASSES = 7
TASK = 'fer2013'
USE_PRETRAINED = True


my_model, input_size = initialize_model(model_name=MODEL_NAME, feature_extract=FEATURE_EXTRACT,
                                        num_classes=NUM_CLASSES, task=TASK, use_pretrained=USE_PRETRAINED)


# Define the optimizer
optimizer = optim.Adam(my_model.fc.parameters(), lr=1e-3)
