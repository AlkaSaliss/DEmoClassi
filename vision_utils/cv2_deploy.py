import imutils
import cv2
import torch
from vision_utils.custom_architectures import PretrainedMT, SepConvModel, SepConvModelMT
from vision_utils.custom_torch_utils import initialize_model
from emotion_detection.evaluate import predict_fer
from multitask_rag.evaluate import predict_utk
import numpy as np
import os
import pathlib
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# default path to a saved model for race, age and gender prediction
saved_weight_utk = '/media/sf_Documents/COMPUTER_VISION/DEmoClassi/' \
                   'multitask_rag/checkpoints/vgg_model_21_val_loss=4.139335.pth'

# default path to a saved model for emotion prediction
saved_weight_fer = '/media/sf_Documents/COMPUTER_VISION/DEmoClassi/' \
                   'emotion_detection/checkpoints/vgg_model_173_val_accuracy=0.6447478.pth'


# paths to the caffe model files for detecting faces using opencv
package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_binaries = os.path.join(package_path, 'cv2_dnn_model_files')
path_caffe_model = os.path.join(path_binaries, 'res10_300x300_ssd_iter_140000.caffemodel')
path_proto = os.path.join(path_binaries, 'deploy.prototxt.txt')


def dict_prob_to_list(dict_probs):
    """
    utility function for converting a dictionary of labels with their probabilities
     into two lists of labels and probs resp.
    """
    items = list(dict_probs.items())
    return [item[0] for item in items], [item[1] for item in items]


def plot_to_array(x, y, color):
    """Utility function for ploting predicted probabilities as bar plots"""
    fig = plt.figure(figsize=(2, 2))
    fig.add_subplot(111)
    # fig.tight_layout(pad=0)
    plt.barh(x, y, color=color)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data


def predict_from_frame(net, frame, model_utk, model_fer, transfer_learn, display_probs=False):
    """
    Makes emotion, gender, age and race prediction from a frame and plot the results in the frame to display
     using opencv
    :param net: opencv face detector
    :param frame: numpy array representing the image from hich to detect face and make prediction
    :param model_utk: pytorch model for predicting race, age and gender
    :param model_fer: pytorch model for predicting emotion
    :param transfer_learn: whether we are using a pretrained model (`resnet` or `vgg`)
    :param display_probs: True or False, whether to plot the predicted probabilities for each class
    :return:
    """

    frame = imutils.resize(frame, width=600, height=600)

    # Prepare the opencv face detector
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # iterate through the detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # if the model has detected face with at least 50% confidence
        # get the bounding box of the face and plot it
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # cv2.rectangle(frame, (startX - 25, startY - 50), (endX + 25, endY + 25), (0, 255, 0), 3)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)

            # do age, gender and race prediction
            # face = frame[startY-25: endY+25, startX-25: endX+25]
            face = frame[startY: endY, startX: endX]
            age, gender, gender_lab, race, race_lab = predict_utk(face, model_utk)

            # do emotion prediction
            try:
                # gray = cv2.cvtColor(frame[startY - 50: endY + 25, startX - 25: endX + 25],
                #                     cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(frame[startY-15: endY+15, startX-15: endX+15],
                                    cv2.COLOR_BGR2GRAY)
                emotion_probs, emotion = predict_fer(gray, model_fer, transfer_learn)

                # add predicted labels to image
                text = f"{race_lab} {emotion} {gender_lab} - Age: {age:.1f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, (startX - 26, startY - 55), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                if display_probs:
                    # plot the predicted probabilities as bar plots
                    emotion_labels, emotion_proba = dict_prob_to_list(emotion_probs)
                    gender_labels, gender_proba = dict_prob_to_list(gender)
                    race_labels, race_proba = dict_prob_to_list(race)

                    emotion_plot = plot_to_array(emotion_labels, emotion_proba, 'g')
                    race_plot = plot_to_array(race_labels, race_proba, 'r')
                    gender_plot = plot_to_array(gender_labels, gender_proba, 'b')

                    cv2.imshow('Emotion', emotion_plot)
                    cv2.imshow('Gender', gender_plot)
                    cv2.imshow('Race', race_plot)
                    plt.close('all')

            except:
                pass
    return frame


parser = argparse.ArgumentParser(description='Realtime emotion, age, gender and race prediction '
                                             'using pytorch and opencv')
parser.add_argument('--emotion_model_weight', type=str, default=saved_weight_fer,
                    help='Path to the trained pytorch model for emotion classification')
parser.add_argument('--demogr_model_weight', type=str, default=saved_weight_utk,
                    help='Path to the trained pytorch model for race, age and gender prediction')
parser.add_argument('--type_emotion_model', type=str, default='vgg',
                    help='`resnet`, `vgg` or `sep_conv` type of model to use for inference')
parser.add_argument('--type_demog_model', type=str, default='vgg',
                    help='`resnet`, `vgg` or `sep_conv` type of model to use for inference')
parser.add_argument('--source', type=str, default='stream',
                    help='`stream`, `image` or `video`: whether to make predictions from streaming (webcam),'
                         'from a video file or from an image')
parser.add_argument('--file', type=str, default=None,
                    help='Path to an image or video file to make predictions from')
parser.add_argument('--display_probs', type=str, default='true',
                    help='true or false, whether to plot the predicted probabilities for each class')
predict_args = parser.parse_args()


def main(args, net):
    emotion_model = args.emotion_model_weight
    demogr_model = args.demogr_model_weight
    type_emotion_model = args.type_emotion_model
    type_demog_model = args.type_demog_model
    from_source = args.source
    source_file = args.file
    if from_source in ['image', 'video'] and source_file is None:
        raise ValueError('You must provide a path to an image/video in order to make predcitions from file')
    display_probs = True if args.display_probs == 'true' else False

    # load emotion detection model
    if type_emotion_model in ['resnet', 'vgg']:
        model_fer, _ = initialize_model(type_emotion_model, False, use_pretrained=False)
        transfer_learn = True
    else:
        model_fer = SepConvModel()
        transfer_learn = False

    model_fer.load_state_dict(torch.load(emotion_model, map_location='cpu'))

    # load age-race-gender prediction model
    if type_demog_model in ['resnet', 'vgg']:
        model_utk = PretrainedMT(type_demog_model, feature_extract=False, use_pretrained=False)
    else:
        model_utk = SepConvModelMT()

    model_utk.load_state_dict(torch.load(demogr_model, map_location='cpu'))

    # make prediction from an input image
    if from_source == 'image':
        frame = cv2.imread(source_file)
        frame = predict_from_frame(net, frame, model_utk, model_fer, transfer_learn, display_probs)
        parent, f_name = str(pathlib.Path(source_file).parent), pathlib.Path(source_file).name
        cv2.imwrite(os.path.join(parent, f_name+'predicted.jpg'), frame)
        cv2.imshow('Face Detector', frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()

    # make prediction from an input video or from camera stream
    else:
        if from_source == 'video':
            vs = cv2.VideoCapture(source_file)
        else:
            vs = cv2.VideoCapture(0)

        while vs.isOpened():
            ret, frame = vs.read()
            frame = predict_from_frame(net, frame, model_utk, model_fer, transfer_learn, display_probs)
            cv2.imshow('Face Detector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        vs.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # opencv face detector
    cv2_facenet = cv2.dnn.readNetFromCaffe(path_proto, path_caffe_model)

    # start detection and prediction
    main(predict_args, cv2_facenet)
