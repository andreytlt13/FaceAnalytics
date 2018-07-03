import cv2

CASCADE_CLASSIFIER = cv2.CascadeClassifier('../../video_processing/haar_cascade_face.xml')


class FaceDetector:

    @staticmethod
    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return CASCADE_CLASSIFIER.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
