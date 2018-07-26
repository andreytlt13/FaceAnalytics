import unittest


class TestCamera(unittest.TestCase):
    def test_camera_has_picture(self):
        cam = cv2.VideoCapture("resources/SampleVideo_1280x720_1mb.mp4")
        success, frame = cam.get_frame()

        self.assertTrue(success)
        self.assertIsNotNone(frame)

    def test_camera_disconnected(self):
        cam = cv2.VideoCapture("no_camera_url")

        self.assertRaises(RuntimeError, cam.get_frame)
