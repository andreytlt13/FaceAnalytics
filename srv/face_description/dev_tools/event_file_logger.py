from datetime import datetime

LOG_PATH = '/tmp/faces_log.txt'

log_time = 0


def log(face_feature_map):
    global log_time
    current_time = datetime.now()
    tens_of_secs_since_last_log = (current_time.timestamp() - log_time) / 10
    # log every 10 seconds
    if tens_of_secs_since_last_log >= 1:
        out = open(LOG_PATH, 'a+')
        for i in range(len(face_feature_map['name'])):
            msg = ''
            for v in face_feature_map.values():
                msg += repr(v[i]) + ', '
            msg = msg[:-2]  # truncate last ', '

            log_time = current_time.timestamp()
            out.write('\n' + msg)
        out.close()
