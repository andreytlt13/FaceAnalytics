import srv.common.api as face_recognition

UNKNOWN = 'Unknown'

andrey_image = face_recognition.load_image_file("srv/video_processing/photo/andrey.jpg")
andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]
simon_image = face_recognition.load_image_file("srv/video_processing/photo/simon.jpg")
simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

known_face_encodings = [
    andrey_face_encoding,
    simon_face_encoding
]
known_face_names = [
    'Andrey',
    'Simon'
]


def recognize_faces(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = UNKNOWN

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    return face_locations, face_names
