import hashlib


def hash_string(camera_url):
    return str(hashlib.sha1(str(camera_url).encode('utf-8')).hexdigest())
