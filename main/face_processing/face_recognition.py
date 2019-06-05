import glob
from face_processing import dlib_api

def recognize_face(best_detected_face, known_face_encodings, known_face_names):
    face_encodings = dlib_api.face_encodings(best_detected_face)

    name = "Unknown"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = dlib_api.compare_faces(known_face_encodings, face_encoding)

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
    return name

def load_known_face_encodings(db_path):
    print('[INFO] loading db faces ...')
    photos = glob.glob(db_path + '/*.png')
    known_face_encodings = []
    known_face_names = []

    for photo in photos:
        image = dlib_api.load_image_file(photo)
        title = photo.split('/')[-1].split('.')[0]
        face_encoding = dlib_api.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(title)

    return known_face_encodings, known_face_names


