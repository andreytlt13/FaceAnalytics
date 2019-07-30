import json


def get_result(vs, tested_vid):
    tr_objects = vs.trackableObjects
    vid_name = tested_vid.split('.')[0]
    try:
        with open('{}.json'.format(vid_name)) as json_file:
            data = json.load(json_file)
        persons = data['persons']
        visible_faces = data['visible_faces']
    except:
        vs.test_info.append('[WARNING!!!] json description for a tested video doesnt exist!\n')
        print('[WARNING!!!] json description for a tested video doesnt exist!')
        persons = 0
        visible_faces = 0

    detected_persons = 0
    detected_faces = 0
    recognized_faces = 0
    if len(tr_objects.items()) > 0:
        detected_persons = len(tr_objects)
        for tr_indx, tr_obj in tr_objects.items():

            if tr_obj.names[0] is not None:
                recognized_faces += 1
            if len(tr_obj.face_seq):
                detected_faces += 1

    vs.test_info \
        .append('[ACCURACY RESULT INFO] actual persons: {}, detected_persons: {}\n[ACCURACY RESULT INFO] actual visible faces: {}, detected_faces: {}, recognized: {}\n'\
        .format(persons, detected_persons, visible_faces, detected_faces, recognized_faces))