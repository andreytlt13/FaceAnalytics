import time
import datetime
import os
import cv2

from video_processing import VideoStream
from common import config_parser


CONFIG = config_parser.parse()

tests_dir = os.path.join(CONFIG["root_path"], CONFIG["tests_dir"])


camera_url = 'rtsp://admin:admin@10.101.1.221:554/ch01/0'

tmp_name = camera_url.replace('/', '_')

save_dir = os.path.join(tests_dir, 'cctv_records/{}'.format(tmp_name))
out_vid_path = os.path.join(save_dir, 'rtsp_record_{}_{}.mp4'.format(tmp_name, datetime.datetime.now()))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

vs = VideoStream(camera_url)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(out_vid_path, fourcc, vs.fps, (vs.W, vs.H))


while True:
    frame = vs.procees_stream()
    out_video.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out_video.release()

