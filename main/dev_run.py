import time
import datetime
import os
import cv2

from video_processing import VideoStream
from common import config_parser

import logging
from main.tests import performance_analysis, accuracy_analysis

CONFIG = config_parser.parse()

log_ = False
analyze_log_ = False
save_res_ = False

stream_ = False

tests_dir = os.path.join(CONFIG["root_path"], CONFIG["tests_dir"])
logs_dir = os.path.join(tests_dir, 'logs')

if stream_:
    camera_url = 'rtsp://user:Hneu74k092@10.101.106.104:554/live/main'
    # camera_url = 0
    tmp_name = camera_url.replace('/', '_')
    vs = VideoStream(camera_url)

else:

    # camera_url = 'vlc_record_2019_05_30_12h50m55s.mp4' #kate_andrey
    camera_url = 'vlc-record-2019-06-04-14h03m36s.mp4' #andrey #frame_out_wh = (600, 450)
    # camera_url = 'vlc-record-2019-06-10-14h42m31s.mp4' #varya_daria
    # camera_url = 'vlc-record-2019-05-27-14h45m56s.mp4' #simon_artemy
    # camera_url = 'vlc-record-2019-05-24-13h45m52s.mp4' #simon_dmitry
    # camera_url = 'vlc_record_2019_05_24_15h29m07s.mp4' #simon_walking_nikita
    # camera_url = 'vlc-record-2019-07-01-16h31m45s.mp4' #andrey_sitting_0
    # camera_url = 'vlc-record-2019-07-01-16h32m15s.mp4' #andrey_sitting_1
    # camera_url = 'vlc-record-2019-07-01-17h38m49s.mp4' #sveta
    # camera_url = 'videoplayback.mp4'
    camera_url = 'andrey_vitya.mp4'
    camera_url = 'vitya.mp4'

    # cctv videos
    # camera_url = '123317_cut_5_persons.mp4' #5_persons
    # camera_url = '100359_4_persons.mp4' #4_persons
    # camera_url = '154522_3_persons.mp4' #3_persons

    vs = VideoStream(os.path.join(os.path.join(CONFIG["root_path"], CONFIG["tests_dir"]),'testing_videos/'+camera_url))
    # vs = VideoStream(0)

class CustomHandler(logging.StreamHandler):
    def __init__(self):
        super(CustomHandler, self).__init__()

    def emit(self, record):
        messages = record.msg.split('\n')
        for message in messages:
            record.msg = message
            super(CustomHandler, self).emit(record)

if log_:
    # file name for saving time log info
    if stream_:
        time_log_name = 'time_perfomance_stream_{}'.format(datetime.datetime.now())
    else:
        time_log_name = 'time_perfomance_{}'.format(camera_url.split('.')[0])

    # create logger
    logger = logging.getLogger('time_perfomance')
    # log all escalated at and above DEBUG
    logger.setLevel(logging.DEBUG)
    # add a file handler
    handler = logging.FileHandler(os.path.join(logs_dir, '{}.csv'.format(time_log_name)))
    handler.setLevel(logging.DEBUG)

    # create a formatter and set the formatter for the handler.
    formattor = logging.Formatter('%(asctime)s,%(message)s')
    handler.setFormatter(formattor)
    # add the Handler to the logger
    logger.addHandler(handler)

# saving video result [optional]
if stream_:
    save_dir = os.path.join(tests_dir, 'results/{}'.format(tmp_name))
    out_vid_path = os.path.join(save_dir, 'res_{}_{}.mp4'.format(tmp_name, datetime.datetime.now()))
else:
    save_dir = os.path.join(tests_dir, 'results/{}'.format(camera_url))
    out_vid_path = os.path.join(save_dir, 'res_{}_{}.mp4'.format(camera_url.split('.')[0], datetime.datetime.now()))

if save_res_:
    # checking folders
    for f in [logs_dir, save_dir]:
        if not os.path.exists(f):
            os.makedirs(f)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_out_wh = (600, 337)
    # out_video = cv2.VideoWriter(out_vid_path, fourcc, 20, frame_out_wh)

    out_video = cv2.VideoWriter('{}.mp4'.format(datetime.datetime.now()), fourcc, 20, frame_out_wh)

t_all = time.monotonic()
while True:
    t_proc_next_frame = time.monotonic()
    img, time_log, tr_objects = vs.process_next_frame
    t_proc_next_frame_elapsed = time.monotonic() - t_proc_next_frame

    if log_:
        time_log.append(t_proc_next_frame)
        time_log = str(time_log).strip('[]')
        logger.info(time_log)

    if img is None:
        break
    cv2.imshow('image', img)

    if save_res_:
        out_video.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if save_res_:
    out_video.release()

print('[TIME LOG] all processed time :', time.monotonic() - t_all)

if analyze_log_:
    performance_analysis.get_result(os.path.join(logs_dir, '{}.csv'.format(time_log_name)), save_dir, face_detection=CONFIG['face_detection'])
    accuracy_analysis.get_result(tr_objects, tested_vid=os.path.join(tests_dir, 'testing_videos/'+camera_url), save_dir=save_dir)