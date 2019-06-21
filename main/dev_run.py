import time
import datetime
import os
import cv2

from video_processing import VideoStream
from common import config_parser

import logging
from main.tests import performance_analysis

CONFIG = config_parser.parse()

camera_url = 'vlc_record_2019_05_30_12h50m55s.mp4' #kate_andrey
# camera_url = 'vlc-record-2019-06-04-14h03m36s.mp4' #andrey
# camera_url = 'vlc-record-2019-06-10-14h42m31s.mp4' #varya_daria

# file name for saving time log info
time_log_name = 'time_perfomance_{}'.format(camera_url.split('.')[0])
tests_dir = os.path.join(CONFIG["root_path"], CONFIG["tests_dir"])
os.makedirs(tests_dir, exist_ok=True)

class CustomHandler(logging.StreamHandler):
    def __init__(self):
        super(CustomHandler, self).__init__()

    def emit(self, record):
        messages = record.msg.split('\n')
        for message in messages:
            record.msg = message
            super(CustomHandler, self).emit(record)

# create logger
logger = logging.getLogger('time_perfomance')
# log all escalated at and above DEBUG
logger.setLevel(logging.DEBUG)
# add a file handler
handler = logging.FileHandler(os.path.join(tests_dir, 'logs/{}.csv'.format(time_log_name)))
handler.setLevel(logging.DEBUG)

# create a formatter and set the formatter for the handler.
formattor = logging.Formatter('%(asctime)s,%(message)s')
handler.setFormatter(formattor)
# add the Handler to the logger
logger.addHandler(handler)

vs = VideoStream(os.path.join(tests_dir,'testing_videos/'+camera_url))

while True:
    t_proc_next_frame = time.monotonic()
    img, time_log = vs.process_next_frame()
    t_proc_next_frame_elapsed = time.monotonic() - t_proc_next_frame

    time_log.append(t_proc_next_frame)
    time_log = str(time_log).strip('[]')
    logger.info(time_log)

    if img is None:
        break
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

performance_analysis.get_result(os.path.join(tests_dir, 'logs/{}.csv'.format(time_log_name)))