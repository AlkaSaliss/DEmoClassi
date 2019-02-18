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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract, num_classes=7, task='fer2013', use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet 34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
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
        """ VGG19_bn
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


def create_summary_writer(model, data_loader, log_dir):
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
        log_dir='../../fer2013/logs', launch_tensorboard=False, patience=10):

    if launch_tensorboard:
        os.system('pkill tensorboard')
        os.system('tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(log_dir))
        os.system("npm install -g localtunnel")
        os.system('lt --port 6006 >> /content/url.txt 2>&1 &')
        os.system('cat /content/url.txt')

    # Get the model, optimizer and dataloaders from script
    model_script = dict()
    with open(path_to_model_script) as f:
        exec(f.read(), model_script)

    model = model_script['my_model']
    optimizer = model_script['optimizer']

    train_loader, val_loader = dataloaders['Training'], dataloaders['PublicTest']

    if launch_tensorboard:
        writer, val_writer = create_summary_writer(model, train_loader, log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.cross_entropy)},
                                            device=device)

    def get_val_loss(engine):
        return -engine.state.metrics['nll']

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

        if launch_tensorboard:
            val_writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
            val_writer.add_scalar('avg_accuracy', avg_accuracy, engine.state.epoch)

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
