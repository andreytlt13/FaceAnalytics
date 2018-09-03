# construct the arguments and parse them
import argparse

from face_description import face_descriptor

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--source', required=False, default='/tmp/faces',
                help='path to output directory')
ap.add_argument('-o', '--output', required=False, default='/tmp/description',
                help='path to output directory')
args = vars(ap.parse_args())

face_descriptor.start(source=args['source'], descriptions_dir=args['output'])
