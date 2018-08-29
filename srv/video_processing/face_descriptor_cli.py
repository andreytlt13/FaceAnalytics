# construct the arguments and parse them
import argparse

from video_processing.face_descriptor import describe

SOURCE = 'source'

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, default='/tmp/faces',
                help="path to output directory")
ap.add_argument("-o", "--output", required=False, default='/tmp/description',
                help="path to output directory")
args = vars(ap.parse_args())

describe(camera_url=0, source=args[SOURCE])
