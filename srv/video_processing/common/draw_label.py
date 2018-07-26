import cv2


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.4, thickness=3):
    cv2.putText(image, label, point, font, font_scale, (54, 255, 81), thickness)