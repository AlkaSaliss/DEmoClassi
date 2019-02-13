# Import
import sys
sys.path.append('../')
from vision_utils.custom_torch_utils import run
from data import data_loader_lambda, BATCH_SIZE, DATA_DIR
import argparse


# Define the constants/hyperparameters
EPOCHS = 10
CHECKPOINT = "../checkpoints"
LOG_INTERVAL = 2
FILE_NAME = 'resnet'
PATH_TO_MODEL_SCRIPT = './model_configs/sep_conv.py'



parser = argparse.ArgumentParser('Train a pytorch model')
parser.add_argument('--data_loader', type=str, default='get_dataloaders_fer48',
                    help='name of the dataloader functions, current choices include get_dataloaders_fer48 and '
                         'get_dataloaders')
parser.add_argument('--resize', nargs='+', type=int, default=None,
                            help='int or 2 or tuple of ints to resize the input image')
parser.add_argument('--hist_eq', type=int, default=1,
                            help='whether to apply hsitogram equalization (1) or not (0)')
parser.add_argument('--chunksize', type=int, default=10000, help='chunksize for reading images from csv')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for train and validation data')
parser.add_argument('--path_to_model_script', type=str, default=PATH_TO_MODEL_SCRIPT,
                    help='path to the script containing model and optimizer definition')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training iterations')
parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                    help='root directory containing whether train and valid folders,'
                         ' or path to csv containing raw fer images pixels')
parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT, help='folder to save checkpoints')
parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL, help='Print metrics each log_interval iterations')
parser.add_argument('--file_name', type=str, default=FILE_NAME, help='filename under which to save the checkpoints')
parser.add_argument('--n_saved', type=int, default=2, help='Save the n_saved best models')
parser.add_argument('--log_dir', type=str, default='./', help='directory where to save tensorboard logs')
parser.add_argument('--launch_tensorboard', type=int, default=0,
                    help='whether to start tensorboard automatically (0) or not (1)')
args = parser.parse_args()


if __name__ == '__main__':

    int2bool = {0: False, 1: True}

    print('-----------Creating data loaders---------------------')
    resize = None
    if args.resize:
        resize = tuple([i for i in args.resize])
        if len(resize) == 1:
            resize = resize * 2
    dataloaders = data_loader_lambda[args.data_loader](batch_size=args.batch_size, data_dir=args.data_dir,
                                                      chunksize=args.chunksize, resize=resize,
                                                       hist_eq=int2bool[args.hist_eq])

    print('--------------------print start training--------------------')
    run(args.path_to_model_script, epochs=args.epochs, log_interval=args.log_interval,
        dataloaders=dataloaders, dirname=args.checkpoint_dir, filename_prefix=args.file_name,
        n_saved=args.n_saved, log_dir=args.log_dir, launch_tensorboard=int2bool[args.launch_tensorboard])
