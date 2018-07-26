from datetime import datetime

LOG_PATH = '/tmp/faces_log.txt'

log_time = 0


def log(msg):
    out = open(LOG_PATH, 'a+')
    global log_time
    current_time = datetime.now()
    tens_of_secs_since_last_log = (current_time.timestamp() - log_time) / 10
    # log every 10 seconds
    if tens_of_secs_since_last_log >= 1:
        log_time = current_time.timestamp()
        are_many_people = len(msg.split(',')) > 1
        out.write(
            '\n' + msg + (' were ' if are_many_people else ' was ') + 'there at '
            + current_time.strftime('%H:%M:%S')[0:8]
            + ' on ' + str(current_time.date())
        )
    out.close()
