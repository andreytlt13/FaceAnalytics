import cv2
import numpy as np
import os
import imutils
from main.common import config_parser

CONFIG = config_parser.parse()
CONFIG["root_path"] = os.path.expanduser("~") + CONFIG["root_path"]
# path to face models storage
face_models = os.path.join(CONFIG['root_path'], CONFIG['face_models'])

eye_casc, mouth_casc, nose_casc = CONFIG['face_cascades'].split(',')

eye_cascade = cv2.CascadeClassifier(os.path.join(face_models, eye_casc))
mouth_cascade = cv2.CascadeClassifier(os.path.join(face_models, mouth_casc))
nose_cascade = cv2.CascadeClassifier(os.path.join(face_models, nose_casc))


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus measure,
    # which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def check_models(list_models):
    for indx, cascade in enumerate(list_models):
        if (cascade.empty()):
            print(indx, 'file couldnt load, check out models paths')


def cascade_detection(img):
    # cascades detection: eyes, mouth, noze
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascades_on_face = 0
    roi_gray = gray.copy()
    roi_color = img.copy()
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)
    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cascades_on_face += 1

    mouth = mouth_cascade.detectMultiScale(roi_gray, 1.8, 11)
    if len(mouth) > 0:
        for (mx, my, mw, mh) in mouth[:1]:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
            cascades_on_face += 1

    nose = nose_cascade.detectMultiScale(roi_gray, 1.8, 11)
    if len(nose) > 0:
        for (nx, ny, nw, nh) in nose[:1]:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)
            cascades_on_face += 1

    img = roi_color.copy()
    print('found cascades_on_face:', cascades_on_face)
    return img, cascades_on_face


def select_best_face(faces_sequence_for_person):
    # get the best face from face sequence for current person
    # check_models([eye_cascade, mouth_cascade, nose_cascade])
    fm_faces = []
    casc_faces = []
    for face_indx, face_image in enumerate(faces_sequence_for_person):
        if all(face_image.shape) > 0:
            image = face_image.copy()

            try:
                image = imutils.resize(image, width=300)
            except:
                image = face_image.copy()

            # count cascades face detection
            cascaded_im, cascades_on_face = cascade_detection(image)
            casc_faces.append(cascades_on_face)

    # if there is more than one face with max detected cascades
    # select the best one via max of variance_of_laplacian
    max_cascaded_faces_indxs = np.argwhere(casc_faces == np.amax(casc_faces))
    max_imgs = [faces_sequence_for_person[int(i)] for i in max_cascaded_faces_indxs]

    for face_indx, face_im in enumerate(max_imgs):
        if all(face_image.shape) > 0:
            image = face_im.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm_faces.append(variance_of_laplacian(gray))
        else:
            fm_faces.append(0)
    best_face_idx = np.argmax(fm_faces)
    best_face_img = max_imgs[best_face_idx]

    best_face_img = imutils.resize(best_face_img, width=300)

    return best_face_img
