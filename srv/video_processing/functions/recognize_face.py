import cv2

import srv.common.api as face_recognition
from srv.video_processing.common.log_faces import log

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
    result = {}

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Unknown'

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # log results
    people_names = ''
    for name in face_names:
        people_names += name + ', '
    people_names = people_names[:-2]  # truncate last ', '
    result['people'] = people_names
    log(people_names)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if face_names[0] == 'Unknown':
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (right + 6, top - 6), font, 1.0, (0, 0, 255), 2)

        else:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (right + 6, top - 6), font, 1.0, (0, 255, 0), 2)

    return frame, result
