#!/usr/bin/env python

from datetime import datetime

import cv2
import dlib
import flask
import imutils
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner

import srv.common as face_recognition
from srv.camera_stream.opencv_read_stream import Camera
from srv.models import inception_resnet_v1

LOG_PATH = '/tmp/faces_log.txt'


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


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


andrey_image = face_recognition.load_image_file("srv/webapp/photo/andrey.jpg")
andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]

simon_image = face_recognition.load_image_file("srv/webapp/photo/simon.jpg")
simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

known_face_encodings = [
    andrey_face_encoding,
    simon_face_encoding
]

known_face_names = [
    'Andrey',
    'Simon'
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

log_time = 0

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template(
        'index.html',
        img_path='/static/images/noise.jpg'
    )


@app.route('/analyse', methods=['POST'])
def analyse():
    return flask.render_template(
        'index.html',
        img_path='video_stream?camera_url=' + flask.request.form.get('camera_url')
    )


@app.route('/video_stream', methods=['GET'])
def video_stream():
    try:
        camera_url = int(flask.request.args.get('camera_url'))
    except:
        camera_url = flask.request.args.get('camera_url')
    return flask.Response(
        generate_stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_stream(camera_url):
    sess, age, gender, train_mode, images_pl = load_network(
        'srv/models')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "srv/models/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)
    img_size = 160

    process_this_frame = True

    # Below is the output of calibrate.py script
    DIM = (1280, 720)
    K = np.array(
        [[601.406657865378, 0.0, 714.6361088321798], [0.0, 605.7953276079065, 316.1816796984329], [0.0, 0.0, 1.0]])
    D = np.array([[0.05540844317604619], [-0.8784316408407845], [2.7661761909290625], [-1.5287598287880562]])

    #  Map fish eye image into flat one
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

    while True:
        _, frame = Camera(camera_url).get_frame()

        rotated_frame = imutils.rotate(frame, 15)
        undistorted_frame = cv2.remap(rotated_frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)

        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_RGB2GRAY)
        img_h, img_w, _ = np.shape(frame)

        detected = detector(undistorted_frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        for i, d in enumerate(detected):
            faces[i, :, :, :] = fa.align(undistorted_frame, gray, detected[i])

        if len(detected) > 0:
            ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

        for i, d in enumerate(detected):
            label = "{}, {}".format(int(ages[i]), "Female" if genders[i] == 0 else "Male")
            draw_label(undistorted_frame, (d.left(), d.bottom()), label)

        small_frame = cv2.resize(undistorted_frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
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
            log_msg_builder = ''
            for name in face_names:
                log_msg_builder += name + ', '
            log_msg_builder = log_msg_builder[:-2]  # truncate last ', '
            log_faces(log_msg_builder)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if face_names[0] == 'Unknown':
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(undistorted_frame, name, (right + 6, top - 6), font, 1.0, (0, 0, 255), 2)

            else:
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(undistorted_frame, name, (right + 6, top - 6), font, 1.0, (0, 255, 0), 2)

        _, img_encoded = cv2.imencode('.jpg', undistorted_frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def log_faces(msg):
    out = open(LOG_PATH, 'a+')
    global log_time
    current_time = datetime.now()
    tens_of_secs_since_last_log = (current_time.timestamp() - log_time) / 10
    # log every 10 seconds
    if tens_of_secs_since_last_log >= 1:
        log_time = current_time.timestamp()
        are_many_people = len(msg.split(',')) > 1
        out.write(
            '\n' + msg + (' were ' if are_many_people else ' was ') + 'there at '
            + current_time.strftime('%H:%M:%S')[0:8]
            + ' on ' + str(current_time.date())
        )
    out.close()


@app.route('/text_stream', methods=['GET'])
def text_stream():
    try:
        faces = open(LOG_PATH, 'r')
    except IOError:
        faces = open(LOG_PATH, 'w+')

    objects_info = faces.readlines()
    faces.close()

    if not objects_info:
        msg = 'Logging started...'
    else:
        msg = objects_info[-1]

    return flask.Response(
        msg,
        mimetype='text/xml'
    )


def run():
    app.run(port=9090, debug=True)
