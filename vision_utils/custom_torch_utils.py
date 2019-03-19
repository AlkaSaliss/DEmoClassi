# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite import handlers
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import glob
import shutil
import numpy as np


def set_parameter_requires_grad(model, feature_extracting):
    """
    Utility fucntion for setting parameters gradient collection to true (finetuning) or false (features extraction)
    :param model: pytorch nn.Module class or subclass
    :param feature_extracting: boolean. If True don't collect gradients for the module's parameters
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract, num_classes=7, task='fer2013', use_pretrained=True):

    """
    Instantiate a pytorch model loading eventually pretrained weights from torchvision models zoo

    :param model_name: string indicating model name; valid candidates are `resnet`, `vgg`, `densenet`,
            `squeezenet`, `inception`, `alexnet`
    :param feature_extract: boolean. If True don't collect gradients for the module's parameters
    :param num_classes: number of classes to add for the classification layer
    :param task: string indicating whether we are performing emotion detection or age/gender/race classification.
            in the 1st case (task='fer2013') we just need to replace the classification using the right number of classes.
            In the 2nd case we add a dense layer of size 128, and do not add classification layer as it is a
            multitask task.
    :param use_pretrained: boolean, whether to load pretrained weights trained on imagenet
    :return: a pytorch model and the input size, typically 224 or 229 as different pretrained models may have
            different input image size.
    """

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet 50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if task == 'fer2013':
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.fc = nn.Linear(num_ftrs, 128)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if task == 'fer2013':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, 128)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if task == 'fer2013':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, 128)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if task == 'fer2013':
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, 128)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxiliary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# variable to track validation loss and computing it separately for each handler (checkpoint, early stop, ...)
val_loss = [np.inf]


def create_summary_writer(model, data_loader, log_dir):
    """
    Utility function for creating tensorboard summaries

    :param model: pytorch model
    :param data_loader: a dataloader yielding input and label batches
    :param log_dir: root folder where to log train and validation summaries
    :return: train and validation writers that can be used to track other information later
    """
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'validation'))
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer, val_writer


def run(path_to_model_script, epochs, log_interval, dataloaders,
        dirname='resnet_models', filename_prefix='resnet', n_saved=2,
        log_dir='../../fer2013/logs', launch_tensorboard=False, patience=10,
        resume_model=None, resume_optimizer=None, backup_step=1, backup_path=None,
        n_epochs_freeze=5, n_cycle=5, lr_after_freeze=1e-3):

    """
    Utility function to hide pytorch models training routine.

    :param path_to_model_script: path to the script defining a pytprch model and an optimizer. The script must
            contain at least two variables names `my_model` (for the model to train)
            and `optimizer` (for the optimizer that tracks the parameters of `my_model`).
            Optionally the script may contain a list of pytorch-ignite schedulers, typically learning rate scheduler
            for instance.
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
    :param lr_start:
    :param lr_end:
    :return:
    """

    if launch_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        # os.system('pkill tensorboard')
        # os.system('tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(log_dir))
        # os.system("npm install -g localtunnel")
        # os.system('lt --port 6006 >> /content/url.txt 2>&1 &')
        # os.system('cat /content/url.txt')

    # Get the model, optimizer and dataloaders from script
    model_script = dict()
    with open(path_to_model_script) as f:
        exec(f.read(), model_script)

    model = model_script['my_model']
    optimizer = model_script['optimizer']
    lr_schedulers = model_script.get('lr_schedulers', None)

    if resume_model:
        model.load_state_dict(torch.load(resume_model))
    if resume_optimizer:
        optimizer.load_state_dict(torch.load(resume_optimizer))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    train_loader, val_loader = dataloaders['Training'], dataloaders['PublicTest']

    if launch_tensorboard:
        writer, val_writer = create_summary_writer(model, train_loader, log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # variable that stores val loss
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.cross_entropy)},
                                            device=device)

    if lr_schedulers is not None:
        for sched in lr_schedulers:
            sched.cycle_size = len(train_loader) * n_cycle
            # if lr_start is not None:
            #     sched.start_value = lr_start
            # if lr_end is not None:
            #     sched.end_value = lr_end
            trainer.add_event_handler(Events.ITERATION_STARTED, sched)

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
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
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
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

        global val_loss
        val_loss.append(avg_nll)

        if launch_tensorboard:
            val_writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
            val_writer.add_scalar('avg_accuracy', avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def unfreeze(engine):
        if engine.state.epoch == n_epochs_freeze:
            for param in model.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    optimizer.add_param_group(
                        {'params': param, "lr": lr_after_freeze}
                    )

    def get_val_loss(engine):
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
