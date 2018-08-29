from flask_api import flask_streaming_api
from video_processing import face_descriptor

face_descriptor.describe()
flask_streaming_api.run()
