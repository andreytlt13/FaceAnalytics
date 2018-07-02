#!/usr/bin/env python
from flask import Flask, render_template, Response

from srv.camera.opencv_read_stream import Camera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def generate_stream(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(Camera('config.ini')),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port='9090', debug=True)
