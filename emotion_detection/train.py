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
parser.add_argument('--to_rgb', type=int, default=0,
                            help='whether (1) to convert gray to rgb (repeat the images 3 times) or not (0)')
parser.add_argument('--add_channel_dim', type=int, default=0,
                            help='whether (1) to add a third channel dimension to the '
                                 'array image (to get h*w*c) or not (0)')
parser.add_argument('--normalize', type=int, default=0,
                            help='whether to normalize (1) or not (0), useful for imagenet pretrained models')
parser.add_argument('--hist_eq', type=int, default=1,
                            help='whether to apply histogram equalization (1) or not (0)')
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
parser.add_argument('--patience', type=int, default=10, help='Patience in terms of number of epochs for early stopping')
parser.add_argument('--launch_tensorboard', type=int, default=0,
                    help='whether to start tensorboard automatically (0) or not (1)')
parser.add_argument('--resume_model', type=str, default=None,
                            help='if given, path to an old model checkpoint from which to restore weights')
parser.add_argument('--resume_optimizer', type=str, default=None,
                            help='if given, path to an old optimizer checkpoint from which to restore state from a '
                                 'previous run')
parser.add_argument('--backup_step', type=int, default=1,
                            help='backup current checkpoints in a given directory every backup_step epochs')
parser.add_argument('--backup_path', type=str, default=None,
                            help='path to folder where to backup current checkpoints, typically when training on'
                                 'google colab this is a path to a folder in my google drive '
                                 'so that I can periodically copy my model checkpoints to google drive')
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
                                                       add_channel_dim=int2bool[args.add_channel_dim],
                                                       chunksize=args.chunksize, resize=resize,
                                                       normalize=int2bool[args.normalize],
                                                       to_rgb=int2bool[args.to_rgb],
                                                       hist_eq=int2bool[args.hist_eq])

    print('--------------------print start training--------------------')
    run(args.path_to_model_script, epochs=args.epochs, log_interval=args.log_interval,
        dataloaders=dataloaders, dirname=args.checkpoint_dir, filename_prefix=args.file_name,
        n_saved=args.n_saved, log_dir=args.log_dir,
        launch_tensorboard=int2bool[args.launch_tensorboard], patience=args.patience,
        resume_model=args.resume_model, resume_optimizer=args.resume_optimizer,
        backup_step=args.backup_step, backup_path=args.backup_path)
