# Import
import torch
import tqdm
import numpy as np
from vision_utils.custom_torch_utils import plot_confusion_matrix


def evaluate_model(model, dataloader,
                   title='Confusion matrix',
                   labels_=[0, 1, 2, 3, 4, 5, 6],
                   target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                   normalize=False):

    """
    Function for evaluating a classification model by printing/plotting classification report and confusion matrix

    :param model: a pytorch trained model
    :param dataloader: a pytorch DataLoader object, or any object that yields pytorch tensors
            ready to be used by the model
    :param title: a string to be used as the plot title
    :param labels_: list of integers (0 to number of classes - 1)
    :param target_names: list of strings or ints that describe the labels, must have the same length as `labels_`
    :param normalize: whether to show the actual values or in % for the confusion matrix
    :return:
    """
    y_true = []
    y_pred = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # first, get the predictions
    model.eval()  # set model in evaluation mode
    model = model.to(device)

    with torch.no_grad():
        # Iterate over data.
        for inputs, labels in tqdm.tqdm(dataloader):
            inputs = inputs.to(device, dtype=torch.float32)
            y_true.append(labels)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.append(preds.to('cpu').numpy())

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    # print classification report  and confusion matrix
    plot_confusion_matrix(y_true, y_pred, title, labels_, target_names, normalize)
