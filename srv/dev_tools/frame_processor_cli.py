import argparse
import time

import cv2
from imutils.video import VideoStream

from video_processing.common.draw_label import draw_label
from video_processing.frame_processor import FrameProcessor

RED_COLOR = (0, 0, 255)

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=False, default='models/deploy.prototxt',
                help='path to Caffe deploy prototxt file')
ap.add_argument('-m', '--model', required=False, default='models/res10_300x300_ssd_iter_140000.caffemodel',
                help='path to Caffe pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.75,
                help='minimum probability to filter weak detections')
ap.add_argument('-o', '--output', required=False, default='video_processing/tmp/photo',
                help='path to output directory')
args = vars(ap.parse_args())

processor = FrameProcessor(confidence=args['confidence'], model=args['model'], detected_faces_dir=args['output'],
                           prototxt=args['prototxt'])

# initialize the video stream and allow the camera sensor to warmup
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame, fps, H = processor.process_next_frame(vs)
    info = [
        ('FPS', '{:.2f}'.format(fps.fps())),
    ]

    for (i, (k, v)) in enumerate(info):
        text = '{}: {}'.format(k, v)
        draw_label(image=frame, label=text, point=(10, H - ((i * 20) + 20)), color=RED_COLOR)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF1

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
