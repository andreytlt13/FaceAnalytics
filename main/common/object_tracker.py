import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict, deque


def get_cropped_person(orig_frame, resized_frame, resized_box):
    startX, startY, endX, endY = resized_box

    # person box in original frame size: need to cut face from it
    startY_orig = int(startY / resized_frame.shape[1] * orig_frame.shape[1])
    endY_orig = int(endY / resized_frame.shape[1] * orig_frame.shape[1])
    startX_orig = int(startX / resized_frame.shape[0] * orig_frame.shape[0])
    endX_orig = int(endX / resized_frame.shape[0] * orig_frame.shape[0])

    cropped_person = orig_frame[startY_orig: endY_orig, startX_orig: endX_orig]
    return cropped_person


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid, embeding, rect, img):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = {
            "centroid": centroid,
            "embeding": embeding,
            "rect": rect,
            "img": img
        }
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, orig_frame, resized_frame, trackable_objects, embeding_list):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                img = get_cropped_person(orig_frame, resized_frame, resized_box=rects[i])
                # cv2.imwrite('{}.jpeg'.format(i), img)
                # self.register(inputCentroids[i], embeding_list[i], rects[i])
                self.register(inputCentroids[i], embeding_list[i], rects[i], img)

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectCentroids = [n['centroid'] for n in objectCentroids]

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                ##img crop
                img = get_cropped_person(orig_frame, resized_frame, resized_box=rects[col])
                cv2.imwrite('{}_.jpeg'.format(col), img)
                self.objects[objectID] = {
                    "centroid": inputCentroids[col],
                    "embeding": embeding_list[col],
                    "rect": rects[col],
                    "img": img
                }

                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], embeding_list[col], rects[col], None)


            not_flat = [trackable_objects[x].embeding[0] for x in list(trackable_objects.keys())]
            flat_list = [item for sublist in not_flat for item in sublist]


        M = np.zeros([self.objects.__len__(), trackable_objects.__len__()])

        for k, i in zip(np.arange(0, self.objects.__len__(), 1), self.objects.keys()):
            for m, j in zip(np.arange(0, trackable_objects.__len__(), 1), trackable_objects.keys()):
                M[k, m] = self.person_distance(self.objects[i]['embeding'], trackable_objects[j].embeding[0])

        # return the set of trackable objects
        return self.objects, M

    def person_recognizer(self, new_person_vector, known_person_encodings, known_person_names):
        # new_person_vector = api.human_vector(new_person_image)[0]

        matches = self.compare_persons(known_person_encodings, new_person_vector, tolerance=15)

        name = None

        # Or instead, use the known face with the smallest distance to the new face
        person_distances = self.person_distance(known_person_encodings, new_person_vector)
        filtered_lst = [(x, y) for x, y in enumerate(list(person_distances)) if y < 20]

        if len(filtered_lst) < 1:
            return None, name

        best_match_index = min(filtered_lst)[0]
        # best_match_index = np.argmin(person_distances)
        if matches[best_match_index]:
            name = known_person_names[best_match_index]

        print('linalg.norm the smallest distance result match:{}'.format(name))

        return name

    def person_distance(self, person_encodings, person_to_compare):
        if len(person_encodings) == 0:
            return np.empty((0))
        return np.linalg.norm(person_encodings - person_to_compare, axis=1)

    def compare_persons(self, known_person_encodings, person_encoding_to_check, tolerance):
        print(self.person_distance(known_person_encodings, person_encoding_to_check))
        return list(self.person_distance(known_person_encodings, person_encoding_to_check) <= tolerance)


class TrackableObject:
    def __init__(self, objectID, centroid, embeding, rect, img):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.embeding = [embeding]
        self.names = [None]
        self.name = None,
        # self.face_seq = [None]
        self.face_seq = deque(maxlen=5)
        self.face_emb = [None]
        self.rect = rect
        self.img = img
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False


class TrackableObject2:
    def __init__(self, objectID, centroid, names, name="unknown", stars="0", description="not yet"):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.names = [names]
        self.name = name
        self.stars = stars
        self.description = description,
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False
