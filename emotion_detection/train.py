# Import
import sys
sys.path.append('../')
from custom_torch_utils import run
from data import get_dataloaders, BATCH_SIZE, DATA_DIR
import argparse


# Define the constants/hyperparameters
EPOCHS = 10
CHECKPOINT = "../checkpoints"
LOG_INTERVAL = 2
FILE_NAME = 'resnet'
PATH_TO_MODEL_SCRIPT = './model_configs/resnet_extra_inputs.py'


parser = argparse.ArgumentParser('Train a pytorch model')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for train and validation data')
parser.add_argument('--path_to_model_script', type=str, default=PATH_TO_MODEL_SCRIPT,
                    help='path to the script containing model and optimizer definition')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training iterations')
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

    print('--------------------print start training--------------------')
    run(args.path_to_model_script, epochs=args.epochs, log_interval=args.log_interval,
        dataloaders=dataloaders, dirname=args.checkpoint_dir, filename_prefix=args.file_name,
        n_saved=args.n_saved)
