import glob
from face_processing import dlib_api
import numpy as np
import scipy.misc
from PIL import Image
import cv2
# from scipy.spatial.distance import cosine
# from mtcnn.mtcnn import MTCNN
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
import face_recognition

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
    face_encodings = dlib_api.face_encodings(best_detected_face[:, :, ::-1]) # !!!!!

    names = []
    # dists = []
    for face_encoding in face_encodings:
        face_distances = dlib_api.face_distance(known_face_encodings, face_encoding)

        # # See if the face is a match for the known face(s)
        # matches = dlib_api.compare_faces(known_face_encodings, face_encoding)
        #
        # # # If a match was found in known_face_encodings, just use the first one.
        # # if True in matches:
        # #     first_match_index = matches.index(True)
        # #     name = known_face_names[first_match_index]
        # #     names.append(name)
        # for indx, match in enumerate(matches):
        #     if match:
        #         names.append(known_face_names[indx])
        #         dists.append(face_distances[indx])
        #
        # dists_top = np.array(dists)
        # names = [names[i] for i in dists_top.argsort()[:3]]

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


# VGGFace2 face embeddings approach
def load_face_models():
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # create the detector, using default weights

    # face detector model
    detector = MTCNN()

    # detector = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    return model, detector

# load known face embs from base :
def load_known_face_encodings_vgg(model, detector, db_path):
    # takes photo, extracts face, gets embedding
    # return: lists of face encodings, face names
    print('[INFO] loading db faces ...')
    photos = glob.glob(db_path + '/*.jpg')

    known_face_encodings = get_embeddings(model, detector, photos)
    known_face_encodings = [emb for emb in known_face_encodings]
    known_face_names = [photo.split('/')[-1].split('.')[0] for photo in photos]

    print(len(known_face_encodings), known_face_names)
    return known_face_encodings, known_face_names


# extract a single face from a given photograph
def extract_face(detector, filename, required_size=(224, 224)):
    # load image from file
    pixels = scipy.misc.imread(filename, mode='RGB')

    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(model, detector, filenames):
    # extract faces
    faces = [extract_face(detector, f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # perform prediction
    yhat = model.predict(samples)
    print(yhat.shape)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        # print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        return True, score
    else:
        # print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        return False, score


def recognize_face_vgg(model, best_detected_face, known_face_encodings, known_face_names):
    # input: best detected face image, known_face_encodings, known_face_names
    # gets face image -> gets embedding -> compare with known_face_encodings
    # returns: top 3 names close to analyzed person

    image = Image.fromarray(best_detected_face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    samples = np.asarray([face_array], 'float32')
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)[0]

    matches = []
    names = []
    for n, emb in enumerate(known_face_encodings):
        # print(known_face_names[n])
        result, score = is_match(yhat, known_face_encodings[n])
        if result:
            matches.append(score)
            names.append(known_face_names[n])

    matches = np.array(matches)
    # print(matches, matches.argsort()[:3])
    names = [names[i] for i in matches.argsort()[:3]]
    return names


# if __name__ == '__main__':
#     model, detector = load_face_models()
#     known_face_encodings, known_face_names = load_known_face_encodings_vgg(
#         model,
#         detector,
#         db_path = '/home/ekaterinaderevyanka/experiments/VGGFace')
#
#     best_detected_face = scipy.misc.imread('/home/ekaterinaderevyanka/experiments/VGGFace/andreym_test.jpg', mode='RGB')
#
#     names = recognize_face_vgg(model, best_detected_face, known_face_encodings, known_face_names)
#     print(names)
