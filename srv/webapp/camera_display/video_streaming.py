#!/usr/bin/env python
import cv2
import flask

CascadePath = 'haarcascade_face.xml'
face_cascade = cv2.CascadeClassifier(CascadePath)
video_capture = cv2.VideoCapture(0)

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


def generate_stream():

    while True:
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite('frame.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return flask.Response(generate_stream(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port='9090', debug=True)
