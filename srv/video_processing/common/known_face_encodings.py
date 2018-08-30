import glob

import srv.common as face_recognition


def known_face_encoding(path):
    photo = glob.glob(path + '/*.jpg')
    known_face_encodings = []
    known_face_names = []

    for i in photo:
        image = face_recognition.load_image_file(i)
        title = i.split('/')[-1].split('.')[0]
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(title)

    return known_face_encodings, known_face_names
