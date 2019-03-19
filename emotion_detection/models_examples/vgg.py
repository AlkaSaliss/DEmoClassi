import torch.optim as optim
from vision_utils.custom_torch_utils import initialize_model


MODEL_NAME = 'vgg'
FEATURE_EXTRACT = False
NUM_CLASSES = 7
TASK = 'fer2013'
USE_PRETRAINED = True


my_model, input_size = initialize_model(model_name=MODEL_NAME, feature_extract=FEATURE_EXTRACT,
                                     num_classes=NUM_CLASSES, task=TASK, use_pretrained=USE_PRETRAINED)

# Define the optimizer
optimizer = optim.Adam(
    [
        {"params": my_model.classifier[6].parameters(), "lr": 1e-3},
        {"params": my_model.features.parameters()},
        {"params": my_model.classifier[0].parameters()},
        {"params": my_model.classifier[3].parameters()},
    ],
    lr=1e-6,
)
