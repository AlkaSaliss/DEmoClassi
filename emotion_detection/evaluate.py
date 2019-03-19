# Import
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch
import tqdm
import numpy as np
import itertools


labels_ = [0, 1, 2, 3, 4, 5, 6]
target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def load_model(path_to_model_script, saved_weight):

    model_script = dict()
    with open(path_to_model_script) as f:
        exec(f.read(), model_script)

    model = model_script.get('my_model')
    model.load_state_dict(torch.load(saved_weight, map_location='cpu'))

    return model


def evaluate_model(model, dataloader,
                   title='Confusion matrix',
                   labels_=labels_, target_names=target_names,
                   normalize=False):
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
            labels = labels.view(labels.size()[0]).to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.append(preds.to('cpu').numpy())

    # print classification report
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
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
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
