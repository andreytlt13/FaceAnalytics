#!/usr/bin/env python

from datetime import datetime

import cv2
import dlib
import face_recognition
import flask
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner

from srv.camera_stream.opencv_read_stream import Camera
from srv.models import inception_resnet_v1
from srv.utils.normalize_image import fisheye_to_flat
from srv.video_processing.centroidtracker import CentroidTracker
from srv.video_processing.pedestrians_detector import detect_people

LOG_PATH = '/tmp/faces_log.txt'
last_log_message = ''
detected_regions_count = 1

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
    'srv/models')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "srv/models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=160)


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.4, thickness=3):
    cv2.putText(image, label, point, font, font_scale, (54, 255, 81), thickness)


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
    img_size = 160

    while True:
        _, frame = Camera(camera_url).get_frame()
        img_h, img_w, _ = np.shape(frame)

        frame = fisheye_to_flat(frame)
        people = detect_people(frame, img_w)
        if len(people) > 0:
            for i, (x, y, w, h) in enumerate(people):
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                cropped = frame[y:h, x:w, :]
                crop_h, crop_w, _ = np.shape(cropped)
                print('Detected region: ' + str(crop_w) + ', ' + str(crop_h))
                global detected_regions_count
                cv2.imwrite('/tmp/images/frame' + str(detected_regions_count) + '.jpg', cropped)
                detected_regions_count += 1

                detect_face_features(cropped, frame, img_size, x, y)
        else:
            detect_face_features(frame, frame, img_size, 0, 0)

        detect_person(frame)
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def detect_face_features(frame_area, frame, img_size, body_left, body_bottom):
    detected = detector(frame_area, 1)
    detected_faces_count = len(detected)

    faces = np.empty((detected_faces_count, img_size, img_size, 3))
    for i, d in enumerate(detected):
        faces[i, :, :, :] = fa.align(frame_area, cv2.cvtColor(frame_area, cv2.COLOR_RGB2GRAY), d)

    if detected_faces_count > 0:
        print('we have detected somebody!')

        objects = ct.update(detected)
        ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

        ids = list(objects.keys())
        for i, obj_id in enumerate(ids):
            if i >= detected_faces_count:
                break

            label = "{}, {}, ID={}".format(int(ages[i]), "Female" if genders[i] == 0 else "Male", obj_id)
            draw_label(frame, (detected[i].left() + body_left, detected[i].bottom() + body_bottom), label)
            log_faces(label)


def detect_person(frame):
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
    log_msg_builder = ''
    for name in face_names:
        log_msg_builder += name + ', '
    log_msg_builder = log_msg_builder[:-2]  # truncate last ', '
    log_faces(log_msg_builder)

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

    global last_log_message
    if last_log_message != msg:
        last_log_message = msg
        return flask.Response(
            msg,
            mimetype='text/xml'
        )
    else:
        return flask.Response(
            'Too many similar requests',
            status=429,
            mimetype='text/xml'
        )


def run():
    app.run(port=9090, debug=True)
