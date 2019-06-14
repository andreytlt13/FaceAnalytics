import cv2
from video_processing import VideoStream

# camera_url = 'vlc_record_2019_05_30_12h50m55s.mp4' #kate_andrey
camera_url = 'vlc-record-2019-06-04-14h03m36s.mp4' #andrey
# camera_url = 'vlc-record-2019-06-10-14h42m31s.mp4' #varya_daria


vs = VideoStream('testing_videos/'+camera_url)

while True:
    img = vs.process_next_frame()
    if img is None:
        break
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break