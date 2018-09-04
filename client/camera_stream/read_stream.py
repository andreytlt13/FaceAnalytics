import cv2


class Camera:
    def __init__(self, camera_url) -> None:
        self.camera_id = camera_url
        self.capture = cv2.VideoCapture('127.0.0.1:9090/video_stream?camera_url=0')  # int(self.camera_id)

    def __del__(self):
        self.capture.release()

    def get_frame(self):
        success, frame = self.capture.read()
        if success:
            return success, frame
        else:
            raise RuntimeError(str.format('Camera %d has been disconnected', self.camera_id))
