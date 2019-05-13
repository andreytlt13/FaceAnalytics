import cv2
import numpy as np
import os
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib

face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../models/haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('../models/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('../models/haarcascade_mcs_nose.xml')

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
shape_predictor = '../models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
fa = FaceAligner(predictor, desiredFaceWidth=256)

def align_face(image):
    # load the input image, resize it, and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections

    if rects:
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            try:
                image = imutils.resize(image[y:y + h, x:x + w], width=256)
            except:
                image = image.copy()
            faceAligned = fa.align(image, gray, rect)

            return faceAligned
    else:
        return 'no_face'

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus measure,
    # which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def check_models(list_models):
    for indx, cascade in enumerate(list_models):
        if(cascade.empty()):
            print(indx, 'file couldnt load')

# --- CASCADES METHOD ---
def cascade_detection(img):
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
    return img, cascades_on_face

def select_best_face_cascades(faces_sequence_for_person):
    fm_faces = []
    casc_faces = []

    for face_indx, face_image in enumerate(faces_sequence_for_person):
        if all(face_image.shape) > 0:
            image = face_image.copy()
            image = align_face(image)

            if image != 'no_face':
                # cascades face detection
                cascaded_im, cascades_on_face = cascade_detection(image)
                casc_faces.append(cascades_on_face)
            else:
                casc_faces.append(0)

        else:
            casc_faces.append(0)

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

    return best_face_img
