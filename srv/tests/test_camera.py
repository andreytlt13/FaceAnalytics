import unittest

from srv.camera_stream.opencv_read_stream import Camera


class TestCamera(unittest.TestCase):
    def test_camera_has_picture(self):
        cam = Camera("resources/SampleVideo_1280x720_1mb.mp4")
        success, frame = cam.get_frame()

        self.assertTrue(success)
        self.assertIsNotNone(frame)

    def test_camera_disconnected(self):
        cam = Camera("no_camera_url")

        self.assertRaises(RuntimeError, cam.get_frame)
