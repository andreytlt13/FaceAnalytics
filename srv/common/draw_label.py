import cv2

GREEN = (0, 255, 0)


def draw_label(image, label, point, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, color=GREEN, thickness=2):
    cv2.putText(image, label, point, font, font_scale, color, thickness)
