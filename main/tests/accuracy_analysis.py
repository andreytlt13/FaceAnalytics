import os
import json


def get_result(tr_objects, tested_vid, save_dir):
    vid_name = tested_vid.split('.')[0]
    with open('{}.json'.format(vid_name)) as json_file:
        data = json.load(json_file)

    persons = data['persons']
    visible_faces = data['visible_faces']

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

    result = 'Actual persons: {}, detected_persons: {}\nActual visible faces: {}, detected_faces: {}, recognized: {}'.format(
                persons, detected_persons, visible_faces, detected_faces, recognized_faces)
    print(result)

    with open(os.path.join(save_dir, '{}.txt'.format(save_dir.split('/')[-1].split('.')[0])), 'w') as f:
        f.write(result)

    return result
