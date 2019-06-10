import cv2
from video_processing import VideoStream

camera_url = "local_path"
vs = VideoStream(camera_url)

while True:
    img = vs.process_next_frame()
    if img is None:
        break
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break