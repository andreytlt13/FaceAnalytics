from datetime import datetime

import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner

from srv.models import inception_resnet_v1
from srv.video_processing.common.draw_label import draw_label
from srv.video_processing.object_tracker import CentroidTracker

ct = CentroidTracker()


def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
    return sess, age, gender, train_mode, images_pl


sess, age, gender, train_mode, images_pl = load_network(
    'srv/models'
)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "srv/models/shape_predictor_68_face_landmarks.dat"
)
fa = FaceAligner(predictor, desiredFaceWidth=160)


def detect_face_features(frame_area, frame, img_size, body_left, body_bottom):
    result = {}

    detected = detector(frame_area, 1)
    detected_faces_count = len(detected)

    if detected_faces_count > 0:
        print('we have detected somebody!')

        faces = np.empty((detected_faces_count, img_size, img_size, 3))
        for i, d in enumerate(detected):
            faces[i, :, :, :] = fa.align(frame_area, cv2.cvtColor(frame_area, cv2.COLOR_RGB2GRAY), d)

        objects = ct.update(detected)
        ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

        ids = list(objects.keys())
        for i, obj_id in enumerate(ids):
            if i >= detected_faces_count:
                break

            age_i = int(ages[i])
            gender_i = "Female" if genders[i] == 0 else "Male"
            result.setdefault('id', []).append(obj_id)
            result.setdefault('time', []).append(datetime.now().strftime('%H:%M:%S')[0:8])
            result.setdefault('age', []).append(age_i)
            result.setdefault('gender', []).append(gender_i)

            label = "{}, {}, ID={}".format(age_i, gender_i, obj_id)
            draw_label(frame, (detected[i].left() + body_left, detected[i].bottom() + body_bottom), label)

    return frame_area, result
