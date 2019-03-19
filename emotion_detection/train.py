# Import
import os
import torch
import torch.nn.functional as F
from custom_torch_utils import create_summary_writer
from ignite.metrics import Loss, Accuracy
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.engine import Events
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
from ignite import handlers
import shutil
import tqdm
import glob
import numpy as np


# variable to track validation loss and computing it separately for each handler (checkpoint, early stop, ...)
val_loss = [np.inf]


def run_fer(model, optimizer, epochs, log_interval, dataloaders,
            dirname='resnet_models', filename_prefix='resnet', n_saved=2,
            log_dir='../../fer2013/logs', launch_tensorboard=False, patience=10,
            resume_model=None, resume_optimizer=None, backup_step=1, backup_path=None,
            n_epochs_freeze=5, n_cycle=None, lr_after_freeze=1e-3,
            lr_cycle_start=1e-4, lr_cycle_end=1e-1):

    """
    Utility function to hide pytorch models training routine.

    :param epochs: maximum number of epoch
    :param log_interval: print training info each log_interval iterations
    :param dataloaders: dictionary of data loaders objects, the keys are `Training` and `PublicTesting`
    :param dirname:
    :param filename_prefix:
    :param n_saved:
    :param log_dir:
    :param launch_tensorboard:
    :param patience:
    :param resume_model:
    :param resume_optimizer:
    :param backup_step:
    :param backup_path:
    :return:
    """

    if launch_tensorboard:
        os.makedirs(log_dir, exist_ok=True)

    if resume_model:
        model.load_state_dict(torch.load(resume_model))
    if resume_optimizer:
        optimizer.load_state_dict(torch.load(resume_optimizer))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    train_loader, val_loader = dataloaders['train'], dataloaders['valid']

    if launch_tensorboard:
        writer, val_writer = create_summary_writer(model, train_loader, log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # variable that stores val loss
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.cross_entropy)},
                                            device=device)

    # function to schedule learning rate if needed
    @trainer.on(Events.EPOCH_STARTED)
    def schedule_learning_rate(engine):
        if engine.state.epoch > n_epochs_freeze and n_cycle not in [None, 0]\
                and not getattr(trainer, 'scheduler_set', False):
            scheduler = LinearCyclicalScheduler(optimizer, 'lr', lr_cycle_start,
                                                lr_cycle_end, len(train_loader)*n_cycle)
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
            setattr(trainer, 'scheduler_set', True)

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
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

        if launch_tensorboard:
            writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
            writer.add_scalar('avg_accuracy', avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

        global val_loss
        val_loss.append(avg_nll)

        if launch_tensorboard:
            val_writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
            val_writer.add_scalar('avg_accuracy', avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_STARTED)
    def unfreeze(engine):
        if engine.state.epoch == n_epochs_freeze:
            for param in model.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    optimizer.add_param_group(
                        {'params': param, "lr": lr_after_freeze}
                    )

    def get_val_loss(_):
        global val_loss
        return -val_loss[-1]

    checkpointer = handlers.ModelCheckpoint(dirname=dirname, filename_prefix=filename_prefix,
                                            score_function=get_val_loss,
                                            score_name='val_loss',
                                            n_saved=n_saved, create_dir=True,
                                            require_empty=False, save_as_state_dict=True
                                            )
    earlystop = handlers.EarlyStopping(patience, get_val_loss, trainer)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
                                {'optimizer': optimizer, 'model': model})
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, earlystop)

    # optimizer and model that are in the gdrive, created from a previous run
    if backup_path is not None:
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

    if launch_tensorboard:
        @trainer.on(Events.EPOCH_COMPLETED)
        def add_histograms(engine):
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
    if launch_tensorboard:
        writer.close()
        val_writer.close()
