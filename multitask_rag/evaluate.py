# Import
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from vision_utils.custom_torch_utils import plot_confusion_matrix
from vision_utils.custom_torch_utils import processing_time
from torchvision import transforms


@processing_time
def evaluate_model(model, dataloader,
                   title='Confusion matrix',
                   labels_=[[0, 1], [0, 1, 2, 3, 4]],
                   target_names=[['Male', 'Female'], ['White', 'Black', 'Asian', 'Indian', 'Unknown']],
                   normalize=False):
    """
        Function for evaluating a classification model by printing/plotting classification report and confusion matrix

        :param model: a pytorch trained model
        :param dataloader: a pytorch DataLoader object, or any object that yields pytorch tensors
                ready to be used by the model
        :param title: a string to be used as the plot title
        :param labels_: list  of lists , each sublist is a list of integers (0 to number of classes - 1) representing
                        labels for an output from the model
        :param target_names: list of lists, each sublist is a list of strings or ints that describe the labels,
                            and must have the same length as the corresponding labels it describes from `labels`list
        :param normalize: whether to show the actual values or in % for the confusion matrix
        :return:
        """

    y_age = []
    y_gender = []
    y_race = []
    y_pred_age = []
    y_pred_gender = []
    y_pred_race = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # first, get the predictions
    model.eval()  # set model in evaluation mode
    model = model.to(device)

    with torch.no_grad():
        # Iterate over data.
        for inputs, age, gender, race in tqdm.tqdm(dataloader):
            inputs = inputs.to(device, dtype=torch.float32)
            y_age.append(age)
            y_gender.append(gender)
            y_race.append(race)

            age_pred, gender_pred, race_pred = model(inputs)
            y_pred_age.append(age_pred.to('cpu').numpy())
            _, gender_pred = torch.max(gender_pred, 1)
            _, race_pred = torch.max(race_pred, 1)
            y_pred_gender.append(gender_pred.to('cpu').numpy())
            y_pred_race.append(race_pred.to('cpu').numpy())

    # print classification report
    y_age, y_pred_age = np.concatenate(y_age), np.concatenate(y_pred_age)
    y_gender, y_pred_gender = np.concatenate(y_gender), np.concatenate(y_pred_gender)
    y_race, y_pred_race = np.concatenate(y_race), np.concatenate(y_pred_race)

    print('----------------------- Age prediction -------------------------')
    print(f"Mean Absolute Error {np.abs(y_age - y_pred_age).mean():.4f}")

    print('----------------------- Gender prediction -------------------------')
    plot_confusion_matrix(y_gender, y_pred_gender, title, labels_[0], target_names[0], normalize)

    print('----------------------- Race prediction -------------------------')
    plot_confusion_matrix(y_race, y_pred_race, title, labels_[1], target_names[1], normalize)


def preprocess_utk(image):
    transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    return transf(image).unsqueeze_(0)


def predict_utk(image, model):

    # process image
    image = preprocess_utk(image)

    # prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    image = image.to(device)

    # predict probabilities
    age_pred, gender_pred, race_pred = model(image)
    age_pred = age_pred.detach().to('cpu').numpy()[0][0]
    gender_probs, race_probs = F.softmax(gender_pred, dim=1).detach().to('cpu').numpy()[0],\
                               F.softmax(race_pred, dim=1).detach().to('cpu').numpy()[0]

    # map probabilities to label names
    gender_labs, race_labs = ['Man', 'Woman'], ['White', 'Black', 'Asian', 'Indian', 'Unknown-race']
    gender_label_pred = gender_labs[np.argmax(gender_probs)]
    race_label_pred = race_labs[np.argmax(race_probs)]

    gender = dict(zip(gender_labs, gender_probs))
    race = dict(zip(race_labs, race_probs))

    return age_pred, gender, gender_label_pred, race, race_label_pred
