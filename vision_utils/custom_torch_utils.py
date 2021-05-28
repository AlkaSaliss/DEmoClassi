# Imports
import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter
from ignite.engine.engine import Engine
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import time


def processing_time(func):
    """
    utility function to print execution time of a given function

    :param func: a python function to track the execution time
    :return:
    """
    def func_wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        seconds = time.time() - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print(f"The execution took {h} hours | {m} minutes | {s:.1f} seconds!")
    return func_wrapper


def count_parameters(model):
    """
    Utility function for counting number of trainable and non trainable parameters of a model
    :param model: pytorch model
    :return:
    """
    print(f"Number of trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Number of non-trainable parameters : {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")


def set_parameter_requires_grad(model, feature_extracting):
    """
    Utility function for setting parameters gradient collection to true (finetuning) or false (features extraction)
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
            in the 1st case (task='fer2013') we just need to replace the classification using the right number of
             classes. In the 2nd case we add a dense layer of size 128, and do not add classification layer as it is a
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


class MultiTaskAccuracy(Metric):
    """Custom implementation of pytorch-ignite `Metric` to get multiple metrics from a multitask model"""

    def __init__(self, output_transform=lambda x: x):
        self._output_transform = output_transform
        self.reset()

    def reset(self):
        """
        reset the metric tracking, e.g. at the end of epoch
        :return:
        """
        self._num_correct = [0, 0]
        self._l1_loss_age = 0.0
        self._num_examples = 0

    def update(self, output):
        """
        Compute the number of correct predictions (for gender and race) and the l1 loss (for age) for the current batch
        :param output: tuple, predicted and true outcomes
        :return:
        """
        y_pred, y = output
        y_pred_age, y_pred_gender, y_pred_race = y_pred
        y_age, y_gender, y_race = y

        _, ind_gender = torch.max(y_pred_gender, 1)
        _, ind_race = torch.max(y_pred_race, 1)
        correct_gender = torch.sum(ind_gender == y_gender.data)
        correct_race = torch.sum(ind_race == y_race.data)
        l1_loss = torch.nn.L1Loss()
        l1_loss_age = l1_loss(y_pred_age, y_age)

        # Store the partial (current batch) results to compute the metrcis at epoch level
        self._num_correct[0] += correct_gender.cpu().item()
        self._num_correct[1] += correct_race.cpu().item()
        self._l1_loss_age += l1_loss_age * y_age.shape[0]
        self._num_examples += y_age.shape[0]

    def compute(self):
        """
        Compute the metrics at the end of epoch
        :return:
        """
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._l1_loss_age / self._num_examples,\
               self._num_correct[0] / self._num_examples,\
               self._num_correct[1] / self._num_examples


def my_multi_task_loss(y_pred, y, weights):
    """
    Multitask loss function. Computes the individual loss for each output and combine them in a weighted average
    :param y_pred: predicted output
    :param y: true labels
    :param weights: list of scalars representing the weight to apply to each loss
    :return: the weighted average loss
    """

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
                 batch_size=lambda x: x[0].shape[0], loss_weights=[1/10, 1/0.16, 1/0.44]):
        super(MutliTaskLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self.loss_weights = loss_weights
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
        average_loss = self._loss_fn(y_pred, y, self.loss_weights)

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
                                        loss_weights=[1/10, 1/0.16, 1/0.44],
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

        loss = loss_fn(y_pred, y, loss_weights)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator_multitask(model, metrics={
    'mt_accuracy': MultiTaskAccuracy(),
    'mt_loss': MutliTaskLoss()},
                                          device=None, non_blocking=False,
                                          prepare_batch=_prepare_batch, loss_weights=[1/10, 1/0.16, 1/0.44]):
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
    metrics['mt_loss'].loss_weights = loss_weights
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


def load_model(path_to_model_script, saved_weight):

    model_script = dict()
    with open(path_to_model_script) as f:
        exec(f.read(), model_script)

    model = model_script.get('my_model')
    model.load_state_dict(torch.load(saved_weight, map_location='cpu'))

    return model


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


def plot_confusion_matrix(y_true, y_pred, title='Confusion matrix', labels_=None, target_names=None, normalize=False):
    """
    Utility function for plotting confusion matrix for classification evaluation

    :param y_true: true labels array
    :param y_pred: predicted labels array
    :param title: Title of the confusion matrix plot
    :param labels_: list of unique labels (e.g. in classification with two classes it could be [0, 1)
    :param target_names: names list for unique labels (e.g. in two classes classification it can be ['male', 'female'])
    :param normalize: boolean, whether to print number in confusion matrix as percentage or not
    :return:
    """

    # print classification report
    print(classification_report(y_true, y_pred, labels=labels_, target_names=target_names))

    # plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{cm[i, j]:.4f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    plt.show()


def plot_lr(list_lr, list_steps):
    """
    Utility function to track and plot learning rate
    :param list_lr: list of learning rates tracked duriong training
    :param list_steps: list of steps/iterations
    :return:
    """

    plt.figure(figsize=(8, 6))
    plt.plot(list_steps, list_lr)
    plt.title('Learning rate by training iterations')
    plt.ylabel('learning rate')
    plt.xlabel('Iterations')
    plt.show()
