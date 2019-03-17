# Import
from __future__ import division
import sys
sys.path.append('../')
from vision_utils.custom_torch_utils import create_summary_writer
# from vision_utils.custom_torch_utils import create_summary_writer
from multitask_rag.rag_dataset import get_dataloaders, BATCH_SIZE, DATA_DIR
import argparse
from tqdm import tqdm
from ignite.engine.engine import Engine, Events
from ignite import handlers
import torch
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import os
import glob
import shutil
import numpy as np


class MultiTaskAccuracy(Metric):

    def __init__(self, output_transform=lambda x: x):
        self._output_transform = output_transform
        self.reset()

    def reset(self):
        self._num_correct = [0, 0]
        self._l1_loss_age = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        y_pred_age, y_pred_gender, y_pred_race = y_pred
        y_age, y_gender, y_race = y

        _, ind_gender = torch.max(y_pred_gender, 1)
        _, ind_race = torch.max(y_pred_race, 1)
        correct_gender = torch.sum(ind_gender == y_gender.data)
        correct_race = torch.sum(ind_race == y_race.data)
        l1_loss = torch.nn.L1Loss()
        l1_loss_age = l1_loss(y_pred_age, y_age)

        self._num_correct[0] += correct_gender.cpu().item()
        self._num_correct[1] += correct_race.cpu().item()
        self._l1_loss_age += l1_loss_age * y_age.shape[0]
        self._num_examples += y_age.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._l1_loss_age / self._num_examples,\
               self._num_correct[0] / self._num_examples,\
               self._num_correct[1] / self._num_examples


def my_multi_task_loss(y_pred, y, weights=[0.01, 1, 1]):

    # mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    xe_loss = torch.nn.CrossEntropyLoss()

    y_pred_age, y_pred_gender, y_pred_race = y_pred
    y_age, y_gender, y_race = y

    age_loss = l1_loss(y_pred_age, y_age)
    gender_loss = xe_loss(y_pred_gender, y_gender)
    race_loss = xe_loss(y_pred_race, y_race)

    avg_loss = weights[0]*age_loss + weights[1]*gender_loss + weights[2]*race_loss
    return avg_loss


class MutliTaskLoss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).

    """

    def __init__(self, loss_fn=my_multi_task_loss, output_transform=lambda x: x,
                 batch_size=lambda x: x[0].shape[0]):
        super(MutliTaskLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_loss = self._loss_fn(y_pred, y, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss')

        N = self._batch_size(y)
        self._sum += average_loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    x, y_age, y_gender, y_race = batch
    x, y_age, y_gender, y_race = x.to(device), y_age.to(device), y_gender.to(device), y_race.to(device)

    return x, y_age, y_gender, y_race


def create_supervised_trainer_multitask(model, optimizer, loss_fn=my_multi_task_loss,
                                        device=None, non_blocking=False,
                                        prepare_batch=_prepare_batch):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(_, batch):
        model.train()
        optimizer.zero_grad()
        x, y_age, y_gender, y_race = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y = [y_age, y_gender, y_race]

        y_pred_age, y_pred_gender, y_pred_race = model(x)
        y_pred = [y_pred_age, y_pred_gender, y_pred_race]

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator_multitask(model, metrics={
    'mt_accuracy': MultiTaskAccuracy(),
    'mt_loss': MutliTaskLoss()},
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.

    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            x, y_age, y_gender, y_race = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred_age, y_pred_gender, y_pred_race = model(x)

            y = [y_age, y_gender, y_race]
            y_pred = [y_pred_age, y_pred_gender, y_pred_race]
            return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


PATH_TO_MODEL_SCRIPT = './model_configs/sep_conv_adam.py'
# SRC_DIR = "/media/sf_Documents/COMPUTER_VISION/UTKface_Aligned_cropped/UTKFace"
# DEST_DIR = "/media/sf_Documents/COMPUTER_VISION/UTKface_Aligned_cropped/utk_face_split"
# TRAIN_SPLIT = 0.7
SRC_DIR = None
DEST_DIR = None
TRAIN_SPLIT = None
CHECKPOINT = './checkpoint'
FILE_NAME = 'sep_conv_adam'
LOG_INTERVAL = 2
EPOCHS = 2

# variable to track validation loss and computing it separately for each handler (checkpoint, early stop, ...)
val_loss = [np.inf]


def run(path_to_model_script, epochs, log_interval, dataloaders,
        dirname='resnet_models', filename_prefix='resnet', n_saved=2,
        log_dir='../../fer2013/logs', launch_tensorboard=False, patience=10,
        resume_model=None, resume_optimizer=None, backup_step=1, backup_path=None):

    # if launch_tensorboard:
    #     os.system('pkill tensorboard')
    #     os.system('tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(log_dir))
    #     os.system("npm install -g localtunnel")
    #     os.system('lt --port 6006 >> /content/url.txt 2>&1 &')
    #     os.system('cat /content/url.txt')

    # Get the model, optimizer and dataloaders from script
    model_script = dict()
    with open(path_to_model_script) as f:
        exec(f.read(), model_script)

    model = model_script['my_model']
    optimizer = model_script['optimizer']

    if resume_model:
        model.load_state_dict(torch.load(resume_model))
    if resume_optimizer:
        optimizer.load_state_dict(torch.load(resume_model))

    train_loader, val_loader = dataloaders['train'], dataloaders['valid']

    if launch_tensorboard:
        writer, val_writer = create_summary_writer(model, train_loader, log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = create_supervised_trainer_multitask(model, optimizer, loss_fn=my_multi_task_loss, device=device)
    evaluator = create_supervised_evaluator_multitask(model,
                                                      metrics={'mt_accuracy': MultiTaskAccuracy(),
                                                               'mt_loss': MutliTaskLoss()},
                                                      device=device)

    desc = "ITERATION - loss: {:.3f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter_ = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter_ % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

        if launch_tensorboard:
            writer.add_scalar('training/loss', engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        age_l1_loss, gender_acc, race_acc = metrics['mt_accuracy']
        avg_nll = metrics['mt_loss']
        tqdm.write(
            "Training Results - Epoch: {} Age L1-loss: {:.3f} ** Gender accuracy: {:.3f} "
            "** Race accuracy: {:.3f} ** Avg loss: {:.3f}"
            .format(engine.state.epoch, age_l1_loss, gender_acc, race_acc, avg_nll)
        )

        # if launch_tensorboard:
        #     writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
        #     writer.add_scalar('avg_accuracy', avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        age_l1_loss, gender_acc, race_acc = metrics['mt_accuracy']
        avg_nll = metrics['mt_loss']
        tqdm.write(
            "Validation Results - Epoch: {} Age L1-loss: {:.3f} ** Gender accuracy: {:.3f} **"
            " Race accuracy: {:.3f} ** Avg loss: {:.3f}"
            .format(engine.state.epoch, age_l1_loss, gender_acc, race_acc, avg_nll))

        pbar.n = pbar.last_print_n = 0

        global val_loss
        val_loss.append(avg_nll)

        # if launch_tensorboard:
        #     val_writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
        #     val_writer.add_scalar('avg_accuracy', avg_accuracy, engine.state.epoch)


    def get_val_loss(engine):
        global val_loss
        return -val_loss[-1]

    checkpointer = handlers.ModelCheckpoint(dirname=dirname, filename_prefix=filename_prefix,
                                            score_function=get_val_loss,
                                            # score_function=log_validation_results,
                                            score_name='val_loss',
                                            n_saved=n_saved, create_dir=True,
                                            require_empty=False, save_as_state_dict=True
                                            )
    earlystop = handlers.EarlyStopping(patience, get_val_loss, trainer)
    #
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
                                {'optimizer': optimizer, 'model': model})
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, earlystop)

    # optimizer and model that are in the gdrive, created from a previous run
    original_files = glob.glob(os.path.join(backup_path, '*.pth*'))

    @trainer.on(Events.EPOCH_COMPLETED)
    def backup_checkpoints(engine):
        if backup_path is not None:
            if engine.state.epoch % backup_step == 0:
                # get old model and optimizer files paths so that we can remove them after copying the newer ones
                old_files = glob.glob(os.path.join(backup_path, '*.pth'))

                # get new model and optimizer checkpoints
                new_files = glob.glob(os.path.join(dirname, '*.pth*'))
                if len(new_files) > 0:  # copy new checkpoints from local checkpoint folder to the backup_path folder
                    for f_ in new_files:
                        shutil.copy2(f_, backup_path)

                    if len(old_files) > 0:  # remove older checkpoints as the new ones have been copied
                        for f_ in old_files:
                            if f_ not in original_files:
                                os.remove(f_)

    # if launch_tensorboard:
    #     @trainer.on(Events.EPOCH_COMPLETED)
    #     def add_histograms(engine):
    #         for name, param in model.named_parameters():
    #             writer.add_histogram(name, param.clone().cpu().data.numpy(), engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
    # if launch_tensorboard:
    #     writer.close()
    #     val_writer.close()


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser('Train a pytorch model')
        parser.add_argument('--resize', type=int, default=128,
                            help='int representing height and width for resizing the input image')

        parser.add_argument('--normalize', type=int, default=0,
                            help='whether to normalize (1) or not (0), useful for imagenet pretrained models')

        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for '
                                                                               'train and validation data')
        parser.add_argument('--n_samples', type=int, default=None, help='Number of images to sample,'
                                                                        ' useful for debugging with small sets')
        parser.add_argument('--path_to_model_script', type=str, default=PATH_TO_MODEL_SCRIPT,
                            help='path to the script containing model and optimizer definition')
        parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training iterations')
        parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                            help='root directory containing train, valid and test image folders')
        parser.add_argument('--src_dir', type=str, default=SRC_DIR, help='source directory containing '
                                                                         'raw images if they are '
                                                                         'not already split in train-test-valid')
        parser.add_argument('--dest_dir', type=str, default=DEST_DIR, help='destination where to store train, val '
                                                                           'and test sub-folders after split')
        parser.add_argument('--train_split', type=float, default=None,
                            help='proportion of train split, in the range 0 to 1')
        parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT,
                            help='folder to save checkpoints')
        parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL,
                            help='Print metrics each log_interval iterations')
        parser.add_argument('--file_name', type=str, default=FILE_NAME,
                            help='filename under which to save the checkpoints')
        parser.add_argument('--n_saved', type=int, default=2, help='Save the n_saved best models')
        parser.add_argument('--log_dir', type=str, default='./', help='directory where to save tensorboard logs')
        parser.add_argument('--patience', type=int, default=10,
                            help='Patience in terms of number of epochs for early stopping')
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

    int2bool = {0: False, 1: True}

    print('-----------Creating data loaders---------------------')
    resize = None
    if args.resize is not None:
        resize = (args.resize,) * 2

    dataloaders = get_dataloaders(batch_size=args.batch_size, data_dir=args.data_dir,
                                  resize=resize,
                                  normalize=int2bool[args.normalize],
                                  n_samples=args.n_samples,
                                  src_dir=args.src_dir,
                                  dest_dir=args.dest_dir,
                                  train_split=args.train_split
                                  )

    print('-------------------- start training--------------------')
    run(args.path_to_model_script, epochs=args.epochs, log_interval=args.log_interval,
        dataloaders=dataloaders, dirname=args.checkpoint_dir, filename_prefix=args.file_name,
        n_saved=args.n_saved, log_dir=args.log_dir,
        launch_tensorboard=int2bool[args.launch_tensorboard], patience=args.patience,
        resume_model=args.resume_model, resume_optimizer=args.resume_optimizer,
        backup_step=args.backup_step, backup_path=args.backup_path
        )


if __name__ == '__main__':

    main()
