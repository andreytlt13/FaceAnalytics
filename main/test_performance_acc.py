import time
import datetime
import os
import cv2
import logging

from common import config_parser
from video_processing import VideoStream
from main.tests import performance_analysis, accuracy_analysis


log_ = True
analyze_log_ = True
save_res_ = True
stream_ = False

CONFIG = config_parser.parse()
CONFIG["root_path"] = os.path.expanduser("~") + CONFIG["root_path"]

tests_dir = os.path.join(CONFIG["root_path"], CONFIG["tests_dir"])
logs_dir = os.path.join(tests_dir, 'logs')

if stream_:
    camera_url = 'rtsp://user:Hneu74k092@10.101.106.104:554/live/main'
    camera_url = 0
    tmp_name = str(camera_url).replace('/', '_')
    vs = VideoStream(camera_url)

    # file name for saving time log info
    time_log_name = 'time_perfomance_stream_{}'.format(datetime.datetime.now())
    save_dir = os.path.join(tests_dir, 'results/{}'.format(tmp_name))
    out_vid_path = os.path.join(save_dir, 'res_{}_{}.mp4'.format(tmp_name, datetime.datetime.now()))
else:
    camera_url = 'andrey_vitya.mp4'

    vs = VideoStream(os.path.join(os.path.join(CONFIG["root_path"], CONFIG["tests_dir"]),'testing_videos/'+camera_url))

    # file name for saving time log info
    time_log_name = 'time_perfomance_{}'.format(camera_url.split('.')[0])
    save_dir = os.path.join(tests_dir, 'results/{}'.format(camera_url))
    out_vid_path = os.path.join(save_dir, 'res_{}_{}.mp4'.format(camera_url.split('.')[0], datetime.datetime.now()))

if save_res_:
    # checking folders
    for f in [logs_dir, save_dir, tests_dir]:
        if not os.path.exists(f):
            os.makedirs(f)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_out_wh = (600, 337)
    out_video = cv2.VideoWriter(out_vid_path, fourcc, 20, frame_out_wh)


class CustomHandler(logging.StreamHandler):
    def __init__(self):
        super(CustomHandler, self).__init__()

    def emit(self, record):
        messages = record.msg.split('\n')
        for message in messages:
            record.msg = message
            super(CustomHandler, self).emit(record)

if log_:
    # create logger
    logger = logging.getLogger('time_perfomance')
    # log all escalated at and above DEBUG
    logger.setLevel(logging.DEBUG)
    # add a file handler
    log_file_path = os.path.join(logs_dir, '{}.csv'.format(time_log_name))
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.DEBUG)

    # create a formatter and set the formatter for the handler.
    formattor = logging.Formatter('%(asctime)s,%(message)s')
    handler.setFormatter(formattor)
    # add the Handler to the logger
    logger.addHandler(handler)


# for FPS measuring
frame_counter = 0
t_proc_fr_rate = time.monotonic()
fps_proc_next_frame = 0
all_fps = []

t_all = time.monotonic()
while True:
    t_proc_next_frame = time.monotonic()
    img, time_log = vs.process_next_frame()
    t_proc_next_frame_elapsed = time.monotonic() - t_proc_next_frame

    # for FPS measuring - frame processing
    print('frame_counter:', frame_counter)
    frame_counter += 1
    if (time.time() - t_proc_fr_rate) > 1.0:
        fps_proc_next_frame = frame_counter / (time.monotonic() - t_proc_fr_rate)
        all_fps.append(fps_proc_next_frame)
        print('FPS:{}'.format(fps_proc_next_frame))
        frame_counter = 0
        t_proc_fr_rate = time.monotonic()

    if log_:
        time_log.append(t_proc_next_frame_elapsed)
        time_log.append(fps_proc_next_frame)
        time_log_ = str(time_log).strip('[]')
        logger.info(time_log_)

    cv2.imshow('image', img)

    if save_res_:
        out_video.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if vs.info['TotalFrames'] == 2040:
        break

if save_res_:
    out_video.release()

vs.test_info.append('[TIME LOG] all processed time : {}\n'.format(time.monotonic() - t_all))
vs.test_info.append('[TIME LOG] FPS for frame full processing : min = {}, max = {}, avg = {}\n'.format(min(all_fps),
                                                                                                       max(all_fps),
                                                                                                       sum(all_fps) / len(all_fps)))

if analyze_log_:
    performance_analysis.get_result(vs, os.path.join(logs_dir, '{}.csv'.format(time_log_name)), save_dir, face_detection=CONFIG['face_detection'])
    accuracy_analysis.get_result(vs, tested_vid=os.path.join(tests_dir, 'testing_videos/{}'.format(camera_url)))

with open(os.path.join(save_dir, '{}_result_report.txt'.format(save_dir.split('/')[-1].split('.')[0])), 'w') as f:
    for item in vs.test_info:
        f.write(item)