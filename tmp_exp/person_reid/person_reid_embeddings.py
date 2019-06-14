#!/usr/bin/env python
# coding: utf-8

# A simple human recognition api for re-ID usage

import reid_api as api
import cv2 
import numpy as np
import glob
import os
from matplotlib import pyplot as plt 
import timeit


# STEPS:
# 1. Use detected person image.
# 2. Crop and and resize to 256x128 images. 
# 3. Put image to resnet-50 human feature embedding extractor and get a 128-D feature array. 
# 4. Compare two human by using euclidean distance, the distance means the similarity of two image.


def load_known_encodings(path, analyzing_person_path):
    photos = glob.glob(path + '/*.jpg')
    known_person_encodings = []
    known_person_names = []

    for photo in photos:
        if photo != analyzing_person_path:
            image = cv2.imread(photo)[:,:,::-1]
            person_id = photo.split('/')[-1].split('.')[0]
            human_vector = api.human_vector(image)[0]
            known_person_encodings.append(human_vector)
            known_person_names.append(person_id)

    return known_person_encodings, known_person_names

def person_distance(person_encodings, person_to_compare):
    if len(person_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(person_encodings - person_to_compare, axis=1)

def person_distance_sqrt(person_encodings, person_to_compare):
    dists = []
    for p_encoding in person_encodings:
        if not np.array_equal(p_encoding, person_to_compare):
            dists.append(api.human_distance(p_encoding, person_to_compare))
        else:
            print("!!!")
    match_indx = np.argmin(np.array(dists))
    return match_indx

def compare_persons(known_person_encodings, person_encoding_to_check, tolerance = 0.6):
    return list(person_distance(known_person_encodings, person_encoding_to_check) <= tolerance)


def person_recognizer(new_person_image, known_person_encodings, known_person_names):
    new_person_vector = api.human_vector(new_person_image)[0]

    # sqrt dists calculation
    match_indx = person_distance_sqrt(known_person_encodings, new_person_vector)
    print('person_distance_sqrt result match:{}'.format(known_person_names[match_indx]))
    
    # linalg.norm calculation
    matches = compare_persons(known_person_encodings, new_person_vector, tolerance=10)
    
    name = 'unknown_person'
    
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_person_names[first_match_index]
    
    print('linalg.norm first indx result match:{}'.format(name))

    
    # Or instead, use the known face with the smallest distance to the new face
    person_distances = person_distance(known_person_encodings, new_person_vector)
    best_match_index = np.argmin(person_distances)
    if matches[best_match_index]:
        name_ = known_person_names[best_match_index]
        
    print('linalg.norm the smallest distance result match:{}'.format(name_))

    return known_person_names[match_indx], name, name_




if __name__ == '__main__':

    DB_path = 'dataset/cuhk03/detected'

    person_image_path = 'dataset/cuhk03/detected/00000001_0000_00000002.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000001_0001_00000002.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000000_0000_00000003.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000000_0001_00000001.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000000_0001_00000001.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000005_0001_00000001.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000003_0001_00000000.jpg'
    # person_image_path = 'dataset/cuhk03/detected/00000005_0000_00000000.jpg'


    print('[info] VECTORIZING BASE...')
    t1 = timeit.default_timer()
    known_person_encodings, known_person_names = load_known_encodings(DB_path, person_image_path)
    print('curr time vectorizing: {} sec'.format(round(timeit.default_timer() - t1, 3)))
    print('[info] END VECTORIZING BASE: known_person_names {} {}'.format(len(known_person_names),known_person_names))

    # test images

    fig = plt.figure()
    person_to_recognize = cv2.imread(person_image_path)[:,:,::-1]
    fig.add_subplot(1,3, 1)
    plt.title('new: {}'.format(person_image_path.split('/')[-1].split('.')[0]))
    plt.imshow(person_to_recognize)

    # recognizing person
    print('[info] recognizing current person...')
    t_recog_start = timeit.default_timer()
    matched_person_id_sqrt, matched_person_id_linalg, matched_person_id_linalg_another = person_recognizer(person_to_recognize, known_person_encodings, known_person_names)
    print('[info] current person recognized in time:{}'.format(round(timeit.default_timer() - t_recog_start, 3)))

    # visualization
    matched_person_im = cv2.imread(os.path.join('dataset/cuhk03/detected', matched_person_id_sqrt) + '.jpg')[:,:,::-1]
    fig.add_subplot(1,4, 2)
    plt.title('matched: {}'.format(matched_person_id_sqrt))
    plt.imshow(matched_person_im)

    matched_person_im = cv2.imread(os.path.join('dataset/cuhk03/detected', matched_person_id_linalg) + '.jpg')[:,:,::-1]
    fig.add_subplot(1,4, 3)
    plt.title('matched: {}'.format(matched_person_id_linalg))
    plt.imshow(matched_person_im)

    matched_person_im = cv2.imread(os.path.join('dataset/cuhk03/detected', matched_person_id_linalg_another) + '.jpg')[:,:,::-1]
    fig.add_subplot(1,4, 4)
    plt.title('matched: {}'.format(matched_person_id_linalg_another))
    plt.imshow(matched_person_im)

    plt.show(block=True)