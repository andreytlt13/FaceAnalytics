import cv2
import dlib
import tensorflow as tf
from imutils.face_utils import FaceAligner

from srv.models import inception_resnet_v1

COLOR_DEPTH = 3
FACE_HEIGHT = 160
FACE_WIDTH = 160


def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, FACE_WIDTH, FACE_HEIGHT, COLOR_DEPTH], name='input_image')
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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "srv/models/shape_predictor_68_face_landmarks.dat"
)
fa = FaceAligner(predictor, desiredFaceWidth=FACE_WIDTH)


def detect_faces(frame_area):
    detected = detector(frame_area, 1)
    faces = [fa.align(frame_area, cv2.cvtColor(frame_area, cv2.COLOR_RGB2GRAY), d) for d in detected]

    return detected, faces
