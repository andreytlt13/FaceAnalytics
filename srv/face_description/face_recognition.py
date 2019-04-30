from face_description import dlib_api

# this function was partly created from face_descriptor.py file

def face_recognizer(best_detected_face, known_face_encodings, known_face_names):
    face_locations = dlib_api.face_locations(best_detected_face)
    face_encodings = dlib_api.face_encodings(best_detected_face, face_locations)

    name = "Unknown"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = dlib_api.compare_faces(known_face_encodings, face_encoding)

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
    return name