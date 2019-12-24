import os, math, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from viz_utils import cv_plot_bbox, cv_plot_keypoints, cv_plot_image


class KeyPoints():
    def __init__(self):
        self.ctx = mx.cpu()
        self.detector_name = "ssd_512_mobilenet1.0_coco"
        self.detector = get_model(self.detector_name, pretrained=True, ctx=self.ctx)

        self.detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
        self.detector.hybridize()

        self.estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=self.ctx)
        self.estimator.hybridize()

    def get_keypoints(self, img):
        frame = img.copy()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
        x = x.as_in_context(self.ctx)
        class_IDs, scores, bounding_boxs = self.detector(x)

        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                           output_shape=(128, 96), ctx=self.ctx)
        if len(upscale_bbox) > 0:
            predicted_heatmap = self.estimator(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

            img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                    box_thresh=0.5, keypoint_thresh=0.2)
        else:
            img = frame.copy()
        res_img = cv_plot_image(img)

        return res_img, pred_coords, confidence

if __name__ == '__main__':
    img = cv2.imread('person_img.jpg')

    kp_extractor = KeyPoints()
    result_img, kp, conf = kp_extractor.get_keypoints(img)
    print(kp, conf)
    cv2.imwrite('RESULT.jpg', result_img)