# Import
from custom_torch_utils import ConvModel, run
import torch.optim as optim
from image_dataset import get_dataloaders, BATCH_SIZE, DATA_DIR
import argparse


# Define the constants/hyperparameters
MODEL_NAME = 'resnet'
FEATURE_EXTRACT = True
NUM_CLASSES = 7
USE_PRETRAINED = True
EPOCHS = 10
CHECKPOINT = "..\\checkpoints"
LOG_INTERVAL = 2
FILE_NAME = 'resnet'


# define the model
my_model = ConvModel(model_name=MODEL_NAME, feature_extract=FEATURE_EXTRACT,
                     num_classes=NUM_CLASSES, use_pretrained=USE_PRETRAINED)

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


parser = argparse.ArgumentParser('Train a pytorch model')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for train and validation data')
parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Pretrained model: resnent, vgg, densenet, ...')
parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='Number of classes to classify')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training iterations')
parser.add_argument('--feat_extract', type=int, default=1, help='whether to freeze pretrained weights (1) or not(0)')
parser.add_argument('--use_pretrained', type=int, default=1, help='whether to use pretrained weights (1) or not(0)')
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='root directory containing train and valid folders')
parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT, help='folder to save checkpoints')
parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL, help='Print metrics each log_interval iterations')
parser.add_argument('--file_name', type=str, default=FILE_NAME, help='filename under which to save the checkpoints')
parser.add_argument('--n_saved', type=int, default=2, help='Save the n_saved best models')
args = parser.parse_args()

if __name__ == '__main__':

    print('-----------Creating data loaders---------------------')
    dataloaders = get_dataloaders(args.batch_size, args.data_dir)

    print('----------------Creating a new model-------------------')
    int2bool = {'0': False, 1: True}
    my_model = ConvModel(model_name=args.model_name, feature_extract=int2bool[args.feat_extract],
                         num_classes=args.num_classes, use_pretrained=int2bool[args.use_pretrained])

    print('--------------------print start training--------------------')
    run(my_model, epochs=args.epochs, optimizer=optimizer, log_interval=args.log_interval,
        dataloaders=dataloaders, dirname=args.checkpoint_dir, filename_prefix=args.file_name,
        n_saved=args.n_saved)
