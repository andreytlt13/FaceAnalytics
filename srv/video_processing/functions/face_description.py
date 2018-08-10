import cv2

import srv.common.api as face_recognition

UNKNOWN = 'Unknown'

andrey_image = face_recognition.load_image_file("photo/andrey.jpg")
andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]
simon_image = face_recognition.load_image_file("photo/simon.jpg")
simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

known_face_encodings = [
    andrey_face_encoding,
    simon_face_encoding
]
known_face_names = [
    'Andrey',
    'Simon'
]


def recognize_faces(frame, is_cropped=False):
    result = {}

    scale_factor = 0.25
    if not is_cropped:
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    else:
        small_frame = frame
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = UNKNOWN

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)
        result.setdefault('name', []).append(name)

    scale_back_factor = int(1 / scale_factor)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        if not is_cropped:
            top *= scale_back_factor
            right *= scale_back_factor
            bottom *= scale_back_factor
            left *= scale_back_factor

        font = cv2.FONT_HERSHEY_DUPLEX
        if face_names[0] == UNKNOWN:
            cv2.putText(frame, name, (right + 6, top - 6), font, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(frame, name, (right + 6, top - 6), font, 1.0, (0, 255, 0), 2)

    return frame, result
