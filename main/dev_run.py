import cv2
from video_processing import VideoStream

camera_url = "/Users/andrey/Downloads/Telegram Desktop/vlc_record_2019_06_10_14h42m31s.mp4"
vs = VideoStream(camera_url)

while True:
    img = vs.process_next_frame()
    if img is None:
        break
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break