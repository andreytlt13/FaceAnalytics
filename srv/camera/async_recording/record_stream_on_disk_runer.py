from srv.camera.async_recording.record_stream_on_disk import CameraStreamOnDiskRecorder

CONFIG_FILE_PATH = 'config.ini'

reader = CameraStreamOnDiskRecorder(CONFIG_FILE_PATH)
frames = reader.record_stream()
