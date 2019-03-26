# package import
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import transform
from skimage import exposure
import dlib
import tqdm
from imutils.face_utils import FaceAligner, rect_to_bb
from multiprocessing import Pool
from collections import Counter
import itertools
import time
import warnings
warnings.filterwarnings('ignore')


PATH_DETECTOR = "../shape_predictor_68_face_landmarks.dat"


# function to process one image
def align_and_crop_one(arg=(None, False, None)):

    im, hist_eq, path_detector = arg

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_detector)
    # Instantiate a face aligner object
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # Convert colored image into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # detect the face in the image
    rects = detector(gray, 2)
    if len(rects) > 0:  # if we have at least one face detected
        rects = rects[0]  # consider the 1st face
        face_aligned = fa.align(im, gray, rects)  # align the face

        # cropping the image
        rects = detector(face_aligned, 2)
        if len(rects) > 0:
            rects = rects[0]

            (x, y, w, h) = rect_to_bb(rects)  # get the bounding box coordinates

            face_cropped = transform.resize(face_aligned[y: y + h,
                                            x: x + w, 0], (224, 224))
            if hist_eq:
                return exposure.equalize_hist(face_cropped), 'OK1'
            else:
                return face_cropped, 'OK1'
        else:
            # if no face is found in the aligned face, return just a resized aligned face 224x224
            if hist_eq:
                return exposure.equalize_hist(transform.resize(face_aligned[:, :, 0], (224, 224))), 'OK2'
            else:
                return transform.resize(face_aligned[:, :, 0], (224, 224)), 'OK2'
    else:
        # if no face is found at all in the image, return the resized original image 224x224
        if hist_eq:
            return exposure.equalize_hist(transform.resize(im[:, :, 0], (224, 224))), 'OK3'
        else:
            return transform.resize(im[:, :, 0], (224, 224)), 'OK3'


# function to align faces and compute landmarks
def align_and_crop(list_images, hist_eq=False, class_=0, flag="Training", root_path='', path_detector=None):

    # converting image to 8-bit
    print('--------------Converting images----------------')
    list_images = list(np.uint8(list_images))

    # resizing the  images
    print('--------------Resizing images----------------')
    list_images = [np.repeat(np.expand_dims(im, 2), 3, 2)
                   for im in list_images]

    print('--------------Align and Crop----------------')
    list_args = list(zip(list_images, itertools.repeat(hist_eq), itertools.repeat(path_detector)))
    with Pool() as p:
        results_tuple = p.map(align_and_crop_one, tqdm.tqdm(list_args))

    results1 = [item[0] for item in results_tuple]

    # Saving the data back to disk
    print("-------------Saving processed images to disk-----------")
    os.makedirs(os.path.join(root_path, flag, str(class_)), exist_ok=True)
    path = os.path.join(root_path, flag, str(class_))
    for i, im in tqdm.tqdm(enumerate(results1)):
        plt.imsave(arr=im, fname=os.path.join(path, 'im_'+str(i) + '.png'), cmap='gray')

    results2 = [item[1] for item in results_tuple]

    return dict(Counter(results2))


def extract_pixels(item):
    return np.array(
        [int(i) for i in item.split(' ')]
    ).reshape((48, 48))


def select_list_images(df, class_, flag):
    res = list(df[(df['emotion'] == class_) & (df['Usage'] == flag)]['pixels'].values)
    return [extract_pixels(item) for item in res]

