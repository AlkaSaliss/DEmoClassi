# Import
from vision_utils.custom_torch_utils import create_summary_writer
from vision_utils.custom_torch_utils import create_supervised_trainer_multitask, create_supervised_evaluator_multitask
from vision_utils.custom_torch_utils import my_multi_task_loss, MutliTaskLoss, MultiTaskAccuracy
from vision_utils.custom_torch_utils import processing_time, count_parameters
from vision_utils.custom_torch_utils import plot_lr
import tqdm
from ignite.engine.engine import Events
from ignite import handlers
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
import torch
import os
import glob
import shutil
import numpy as np


# variable to track validation loss and computing it separately for each handler (checkpoint, early stop, ...)
val_loss = [np.inf]


@processing_time
def run_utk(model, optimizer, epochs, log_interval, dataloaders,
            dirname='resnet_models', filename_prefix='resnet', n_saved=2,
            log_dir='../../fer2013/logs', launch_tensorboard=False, patience=10,
            resume_model=None, resume_optimizer=None, backup_step=1, backup_path=None,
            n_epochs_freeze=5, n_cycle=None, lr_after_freeze=1e-3, lr_cycle_start=1e-4,
            lr_cycle_end=1e-1, loss_weights=[1/10, 1/0.16, 1/0.44], lr_plot=True):

    """
    Utility function that encapsulates pytorch models training routine.

    :param model: pytorch model to be trained
    :param optimizer: pytorch optimizer that updates the `model`'s parameters
    :param epochs: maximum number of epoch to train for
    :param log_interval: print training loss each `log_interval` iterations during training
    :param dataloaders: dictionary with `train` and `valid` as keys, the corresponding values being resp. train
            and validation pytorch `DataLoader` objects
    :param dirname: path to the directory where to save model checkpoints during training
    :param filename_prefix: string, name under which to save the model checkpoint file
    :param n_saved: int, save n_saved best model during training
    :param log_dir: optional path to a directory where to write tensorboard logs
    :param launch_tensorboard: boolean, whether to write metrics and histograms using tensorboard
    :param patience: int, number of epochs to wait for before stopping training if no improvement is recorded
    :param resume_model: optional path to checkpoint of trained model to load weights from and continue training
    :param resume_optimizer: optional path to a previous optimizer checkpoint to load state_dict from
    :param backup_step: optional, copy the model checkpoints from `dirname` each `backup_step` epochs,
                        This is useful for me in situation where I train on google colab and want backup my checkpoints
                        to my google drive
    :param backup_path: optional path to backup (copy) model checkpoints to, each `backup_step` epochs.
    :param n_epochs_freeze: after `n_epochs_freeze` unfreeze the model's frozen layers,
                            useful when doing transfer learning
    :param n_cycle: optional int, in terms of number of epochs, to be used for cycle size when doing learning rate
                    scheduling
    :param lr_after_freeze: float, the new learning rate to set after unfreezing the model's layer for finetuning
    :param lr_cycle_start: starting value for learning rate when doing learning rate scheduling
    :param lr_cycle_end: end value for learning rate when doing learning rate scheduling
    :param loss_weights: list of float to be used for weighting model's outputs
    :return:
    """

    count_parameters(model)

    # create the tensorboard log directory if relevant
    if launch_tensorboard:
        os.makedirs(log_dir, exist_ok=True)

    # In case a path of previous model and optimizer checkpoints are provided load weights and state from them
    if resume_model:
        model.load_state_dict(torch.load(resume_model))

    if resume_optimizer:
        optimizer.load_state_dict(torch.load(resume_optimizer))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    # Get the training and validation data loaders
    train_loader, val_loader = dataloaders['train'], dataloaders['valid']

    # create tensorboard writers
    if launch_tensorboard:
        writer, val_writer = create_summary_writer(model, train_loader, log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create trainer and evaluator engines that handle model training and evaluation resp.
    trainer = create_supervised_trainer_multitask(model, optimizer, loss_fn=my_multi_task_loss,
                                                  loss_weights=loss_weights, device=device)
    evaluator = create_supervised_evaluator_multitask(model,
                                                      metrics={'mt_accuracy': MultiTaskAccuracy(),
                                                               'mt_loss': MutliTaskLoss()},
                                                      device=device, loss_weights=loss_weights)

    # function to schedule learning rate if needed
    @trainer.on(Events.EPOCH_STARTED)
    def schedule_learning_rate(engine):
        if engine.state.epoch > n_epochs_freeze and n_cycle not in [None, 0] \
                and not getattr(trainer, 'scheduler_set', False):
            scheduler = LinearCyclicalScheduler(optimizer, 'lr', lr_cycle_start,
                                                lr_cycle_end, len(train_loader) * n_cycle)
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
            setattr(trainer, 'scheduler_set', True)

    # functions to write metrics during training
    desc = "ITERATION - loss: {:.3f}"
    pbar = tqdm.tqdm(
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
        # print metrics on training set
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        age_l1_loss, gender_acc, race_acc = metrics['mt_accuracy']
        avg_nll = metrics['mt_loss']
        tqdm.tqdm.write(
            "Training Results - Epoch: {} Age L1-loss: {:.3f} ** Gender accuracy: {:.3f} "
            "** Race accuracy: {:.3f} ** Avg loss: {:.3f}"
            .format(engine.state.epoch, age_l1_loss, gender_acc, race_acc, avg_nll)
        )
        if launch_tensorboard:
            writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
            writer.add_scalar('age_l1_loss', age_l1_loss, engine.state.epoch)
            writer.add_scalar('gender_accuracy', gender_acc, engine.state.epoch)
            writer.add_scalar('race_accuracy', race_acc, engine.state.epoch)

        # print metrics on validation set
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        age_l1_loss, gender_acc, race_acc = metrics['mt_accuracy']
        avg_nll = metrics['mt_loss']
        tqdm.tqdm.write(
            "Validation Results - Epoch: {} Age L1-loss: {:.3f} ** Gender accuracy: {:.3f} **"
            " Race accuracy: {:.3f} ** Avg loss: {:.3f}"
            .format(engine.state.epoch, age_l1_loss, gender_acc, race_acc, avg_nll))
        global val_loss
        val_loss.append(avg_nll)
        if launch_tensorboard:
            val_writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
            val_writer.add_scalar('age_l1_loss', age_l1_loss, engine.state.epoch)
            val_writer.add_scalar('gender_accuracy', gender_acc, engine.state.epoch)
            val_writer.add_scalar('race_accuracy', race_acc, engine.state.epoch)

        pbar.n = pbar.last_print_n = 0

    # Utility function for unfreezing frozen layer for finetuning
    @trainer.on(Events.EPOCH_STARTED)
    def unfreeze(engine):
        if engine.state.epoch == n_epochs_freeze:
            print('****Unfreezing frozen layers ... ***')
            for param in model.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    optimizer.add_param_group(
                        {'params': param, "lr": lr_after_freeze}
                    )
            count_parameters(model)

    # Function that returns the negative validation loss, useful for saving the best checkpoint at each epoch
    def get_val_loss(_):
        global val_loss
        return -val_loss[-1]

    # callback to save the best model during training
    checkpointer = handlers.ModelCheckpoint(dirname=dirname, filename_prefix=filename_prefix,
                                            score_function=get_val_loss,
                                            # score_function=log_validation_results,
                                            score_name='val_loss',
                                            n_saved=n_saved, create_dir=True,
                                            require_empty=False, save_as_state_dict=True
                                            )

    # callback to stop training if no improvement is observed
    patience *= 2  # because the evaluator is called twice (on training set and validation set)
    earlystop = handlers.EarlyStopping(patience, get_val_loss, trainer)
    #
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
                                {'optimizer': optimizer, 'model': model})
    evaluator.add_event_handler(Events.COMPLETED, earlystop)

    # optimizer and model that are in the backup_path, created from a previous run
    if backup_path is not None:
        original_files = glob.glob(os.path.join(backup_path, '*.pth*'))

    # utility function to periodically copy best model to `backup_path` folder
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

    @trainer.on(Events.COMPLETED)
    def final_backup(_):
        if backup_path is not None:
            new_files = glob.glob(os.path.join(dirname, '*.pth*'))
            if len(new_files) > 0:
                for f_ in new_files:
                    shutil.copy2(f_, backup_path)

    # plot learning rate
    list_lr = [p['lr'] for i, p in enumerate(optimizer.param_groups) if i == 0]
    list_steps = [0]

    @trainer.on(Events.ITERATION_COMPLETED)
    def track_learning_rate(engine):
        if lr_plot is True:
            list_steps.append(engine.state.iteration)
            list_lr.extend([p['lr'] for i, p in enumerate(optimizer.param_groups) if i == 0])

    @trainer.on(Events.EPOCH_COMPLETED)
    def add_histograms(engine):
        if launch_tensorboard:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
    if launch_tensorboard:
        writer.close()
        val_writer.close()

    if lr_plot:
        plot_lr(list_lr, list_steps)