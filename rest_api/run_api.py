import sys

sys.path.append('/Users/andrey/PycharmProjects/FaceAnalytics')
print(sys.path)
from rest_api import streaming_api

streaming_api.run()