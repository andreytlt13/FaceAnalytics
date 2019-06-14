import glob
from face_processing import dlib_api
import numpy as np
import scipy.misc
import cv2

def get_cropped_person(orig_frame, resized_frame, resized_box):
    startX, startY, endX, endY = resized_box

    # person box in original frame size: need to cut face from it
    startY_orig = int(startY / resized_frame.shape[1] * orig_frame.shape[1])
    endY_orig = int(endY / resized_frame.shape[1] * orig_frame.shape[1])
    startX_orig = int(startX / resized_frame.shape[0] * orig_frame.shape[0])
    endX_orig = int(endX / resized_frame.shape[0] * orig_frame.shape[0])

    cropped_person = orig_frame[startY_orig: endY_orig, startX_orig: endX_orig]
    return cropped_person

# dlib face embeddings
def recognize_face(best_detected_face, known_face_encodings, known_face_names):

    face_encodings = dlib_api.face_encodings(best_detected_face) # !!!!!

    print('best_detected_face curr', best_detected_face.shape)
    print('curr face_encodings', face_encodings) # !!!!!

    names = []
    for face_encoding in face_encodings:
        face_distances = dlib_api.face_distance(known_face_encodings, face_encoding)
        face_distances = np.array(face_distances)
        names = [known_face_names[i] for i in face_distances.argsort()[:3]]

    return names, face_encodings

def load_known_face_encodings(db_path):
    print('[INFO] loading db faces ...')
    photos = glob.glob(db_path + '/*.jpg')
    known_face_encodings = []
    known_face_names = []

    for photo in photos:
        image = dlib_api.load_image_file(photo)
        title = photo.split('/')[-1].split('.')[0]
        face_encoding = dlib_api.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(title)

    print('known_face_names:', known_face_names)
    return known_face_encodings, known_face_names
