import cv2
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np

def cv_plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0):
    """Visualize bounding boxes with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        The image with detected results.

    """

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height
    else:
        bboxes *= scale

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        bcolor = [x * 255 for x in colors[cls_id]]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
        if class_name or score:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                        (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                        bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)

    return img

def cv_plot_image(img, scale=1, upperleft_txt=None, upperleft_txt_corner=(10, 100),
                  left_txt_list=None, left_txt_corner=(10, 150),
                  title_txt_list=None, title_txt_corner=(500, 50),
                  canvas_name='demo'):
    """Visualize image with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    scale : float
        The scaling factor of the output image
    upperleft_txt : str, optional, default is None
        If presents, will print the string at the upperleft corner
    upperleft_txt_corner : tuple, optional, default is (10, 100)
        The bottomleft corner of `upperleft_txt`
    left_txt_list : list of str, optional, default is None
        If presents, will print each string in the list close to the left
    left_txt_corner : tuple, optional, default is (10, 150)
        The bottomleft corner of `left_txt_list`
    title_txt_list : list of str, optional, default is None
        If presents, will print each string in the list close to the top
    title_txt_corner : tuple, optional, default is (500, 50)
        The bottomleft corner of `title_txt_list`
    canvas_name : str, optional, default is 'demo'
        The name of the canvas to plot the image

    Examples
    --------

    from matplotlib import pyplot as plt
    ax = plot_image(img)
    plt.show()
    """

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()

    height, width, _ = img.shape
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    if upperleft_txt is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = upperleft_txt_corner
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 3

        cv2.putText(img, upperleft_txt, bottomLeftCornerOfText,
                    font, fontScale, fontColor, thickness)

    if left_txt_list is not None:
        starty = left_txt_corner[1]
        for txt in left_txt_list:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (left_txt_corner[0], starty)
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1

            cv2.putText(img, txt, bottomLeftCornerOfText,
                        font, fontScale, fontColor, thickness)

            starty += 30

    if title_txt_list is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = title_txt_corner
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 3

        for txt in title_txt_list:
            cv2.putText(img, txt, bottomLeftCornerOfText,
                        font, fontScale, fontColor, thickness)
            bottomLeftCornerOfText = (bottomLeftCornerOfText[0] + 100,
                                      bottomLeftCornerOfText[1] + 50)

    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(canvas_name, canvas)
    return canvas

def cv_plot_keypoints(img, coords, confidence, class_ids, bboxes, scores,
                      box_thresh=0.5, keypoint_thresh=0.2, scale=1.0, **kwargs):
    """Visualize keypoints with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    coords : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 2`.
    confidence : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 1`.
    class_ids : numpy.ndarray or mxnet.nd.NDArray
        Class IDs.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    box_thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `box_thresh`
        will be ignored in display.
    keypoint_thresh : float, optional, default 0.2
        Keypoints with confidence less than `keypoint_thresh` will be ignored in display.
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        The image with estimated pose.

    """
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    if isinstance(class_ids, mx.nd.NDArray):
        class_ids = class_ids.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if isinstance(confidence, mx.nd.NDArray):
        confidence = confidence.asnumpy()

    joint_visible = confidence[:, :, 0] > keypoint_thresh
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]

    person_ind = class_ids[0] == 0
    img = cv_plot_bbox(img, bboxes[0][person_ind[:, 0]], scores[0][person_ind[:, 0]],
                        thresh=box_thresh, class_names='person', scale=scale, **kwargs)

    colormap_index = np.linspace(0, 1, len(joint_pairs))
    coords *= scale

    for i in range(coords.shape[0]):
        pts = coords[i]
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                cm_color = tuple([int(x * 255) for x in plt.cm.cool(cm_ind)[:3]])
                pt1 = (int(pts[jp, 0][0]), int(pts[jp, 1][0]))
                pt2 = (int(pts[jp, 0][1]), int(pts[jp, 1][1]))
                cv2.line(img, pt1, pt2, cm_color, 3)
    return img