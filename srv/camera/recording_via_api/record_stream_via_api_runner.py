from srv.camera.recording_via_api.record_stream_via_api import CameraStreamRemoteRecorder

CONFIG_FILE_PATH = 'config.ini'

reader = CameraStreamRemoteRecorder(CONFIG_FILE_PATH)
frames = reader.record_stream()
