import argparse
import socket
import cv2
import imagezmq
import imutils
from imutils.video import VideoStream
import threading
import os


__version__ = '0.1.0'

parser = argparse.ArgumentParser(description='video url or path')
ap = argparse.ArgumentParser()
ap.add_argument("-src", "--source", required=False, help="path to Caffe 'deploy' prototxt file")

args = vars(ap.parse_args())

# class RecordingThread(threading.Thread):
#     def __init__(self, name, camera):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.isRunning = True
#
#         self.cap = camera
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         self.out = cv2.VideoWriter('./static/video.avi', fourcc, 20.0, (640, 480))
#
#     def run(self):
#         while self.isRunning:
#             ret, frame = self.cap.read()
#             if ret:
#                 self.out.write(frame)
#
#         self.out.release()
#
#     def stop(self):
#         self.isRunning = False
#
#     def __del__(self):
#         self.out.release()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', threaded=True)
#

def main(args=None):

    vc = cv2.VideoCapture(args["source"])
    while True:
        ret, frame = vc.read()
        frame = imutils.resize(frame, width=600)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    return "run {}".format(args['source'])


if __name__ == '__main__':
    main(args)


