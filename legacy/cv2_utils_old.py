from imutils.video import VideoStream
import imutils
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from vision_utils.custom_architectures import PretrainedMT
from vision_utils.custom_torch_utils import initialize_model
from emotion_detection.fer_data_utils import SkResize, HistEq, AddChannel, ToRGB
import numpy as np


# Load the trained model for age, gender and race classification
saved_weight_utk = '../multitask_rag/checkpoints/resnet_model_21_val_loss=4.275671.pth'
model_utk = PretrainedMT(model_name='resnet')
model_utk.load_state_dict(torch.load(saved_weight_utk, map_location='cpu'))

# load the trained model for emotion classification
saved_weight_fer = '../emotion_detection/checkpoints/vgg_model_173_val_accuracy=0.6447478.pth'
model_fer, _ = initialize_model('vgg', False, use_pretrained=False)
model_fer.load_state_dict(torch.load(saved_weight_fer, map_location='cpu'))

# load opencv resrnet base face detector
path_caffe_model = '../binary_files/res10_300x300_ssd_iter_140000.caffemodel'
path_proto = '../binary_files/deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(path_proto, path_caffe_model)


def preprocess_utk(image):
    transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return transf(image).unsqueeze_(0)


def preprocess_fer(image, transf_learn):
    if transf_learn:
        transf = transforms.Compose([
            SkResize((224, 224)),
            HistEq(),
            ToRGB(),
            transforms.ToTensor()
        ])
    else:
        transf = transforms.Compose([
            HistEq(),
            AddChannel(),
            transforms.ToTensor()
        ])

    return transf(image).to(torch.float32).unsqueeze_(0)


def predict_utk(image, model):

    # process image
    image = preprocess_utk(image)

    # prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)

    # predict probabilities
    age_pred, gender_pred, race_pred = model(image)
    age_pred = age_pred.detach().to('cpu').numpy()[0][0]
    gender_probs, race_probs = F.softmax(gender_pred, dim=1).detach().to('cpu').numpy()[0],\
                               F.softmax(race_pred, dim=1).detach().to('cpu').numpy()[0]

    # map probabilities to label names
    gender_labs, race_labs = ['Man', 'Woman'], ['White', 'Black', 'Asian', 'Indian', 'Other']
    gender_label_pred = gender_labs[np.argmax(gender_probs)]
    race_label_pred = race_labs[np.argmax(race_probs)]

    gender = dict(zip(gender_labs, gender_probs))
    race = dict(zip(race_labs, race_probs))

    return age_pred, gender, gender_label_pred, race, race_label_pred


def predict_fer(image, model, transf_learn=True):

    # process image
    image = preprocess_fer(image, transf_learn)

    # prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)

    # predict probabilities
    emotion = F.softmax(model(image), dim=1).detach().to('cpu').numpy()[0]
    target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    pred_label = target_names[np.argmax(emotion)]

    emotion_probs = dict(zip(target_names, emotion))

    return emotion_probs, pred_label


vs = VideoStream(src=0).start()
time.sleep(2)

if __name__ == '__main__':

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000, height=1000)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")



        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # # detects faces in the grayscale image
        # rects = detector(gray, 0)

        # if faces found
        # if len(rects) > 0:
        #     for rect in rects:
        #         # get and plot bounding box for each face
        #         (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        #         # cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        #         cv2.rectangle(frame, (bX - 25, bY - 150), (bX + bW + 25, bY + bH + 50), (0, 255, 0), 3)
        #
        #         face = frame[bY: bY + bH, bX: bX + bW, :]
        #         # age, gender, gender_lab, race, race_lab = predict_utk(face, model_utk)
        #         #
        #         # emotion_probs, emotion = predict_fer(gray[bY-10: bY+bH+10, bX-10: bX+bW+10], model_fer)
        #         #
        #         # text = f"{race_lab}({race[race_lab]*100:.0f}%) {emotion}({emotion_probs[emotion]*100:.0f}%)" \
        #         #     f" {gender_lab}({gender[gender_lab]*100:.0f}%) - Age: {age:.1f}"
        #
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         # cv2.putText(frame, text,
        #         #             (bX, bY), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #         # cv2.putText(frame, str(gender),
        #         #             (0, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        #         # cv2.putText(frame, str(race),
        #         #             (0, 75), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        #     # Show the frame with the bounding boxes

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
