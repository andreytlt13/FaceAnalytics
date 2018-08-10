from datetime import datetime

from srv.video_processing.common.draw_label import draw_label
from srv.video_processing.functions.face_feature_detector import detect_faces
from srv.video_processing.object_tracker import CentroidTracker

ct = CentroidTracker()


def detect_gender(frame_area, frame, img_size, body_left, body_bottom, sess, gender, train_mode, images_pl):
    result = {}
    genders = sess.run(gender, feed_dict={images_pl: frame, train_mode: False})




    detected, faces = detect_faces(frame_area, img_size)
    detected_faces_count = len(detected)

    if detected_faces_count > 0:
        print('we have detected somebody!')

        min_horizontal_center = int(min([d.right() - d.left() for d in detected]) / 2)
        min_vertical_center = int(min([d.top() - d.bottom() for d in detected]) / 2)

        objects = ct.update(detected)
        genders = sess.run(gender, feed_dict={images_pl: faces, train_mode: False})

        for i, (obj_id, centroid) in enumerate(objects.items()):
            if i >= detected_faces_count:
                break

            gender_i = "Female" if genders[i] == 0 else "Male"
            result.setdefault('id', []).append(obj_id)
            result.setdefault('time', []).append(datetime.now().strftime('%H:%M:%S')[0:8])
            result.setdefault('gender', []).append(gender_i)

            label = "{}, ID={}".format(gender_i, obj_id)
            draw_label(
                frame,
                (body_left + centroid[0] - min_horizontal_center,
                 body_bottom + centroid[1] + min_vertical_center),
                label
            )

    return frame_area, result
