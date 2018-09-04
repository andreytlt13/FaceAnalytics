import glob

import tensorflow as tf

from config import config_parser
from face_description import dlib_api
from face_description.models import inception_resnet_v1

COLOR_DEPTH = 3
FACE_WIDTH = 160
CONFIG = config_parser.parse_default()


def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, FACE_WIDTH, FACE_WIDTH, COLOR_DEPTH], name='input_image')
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
        print('restore model!')
    else:
        pass
    return sess, age, gender, train_mode, images_pl


def load_known_face_encodings(path):
    photos = glob.glob(path + '/*.jpg')
    known_face_encodings = []
    known_face_names = []

    for photo in photos:
        image = dlib_api.load_image_file(photo)
        title = photo.split('/')[-1].split('.')[0]
        face_encoding = dlib_api.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(title)

    return known_face_encodings, known_face_names
