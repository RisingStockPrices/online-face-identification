import cv2
import numpy as np

COLOR = [list(np.random.random(size=3) * 256) for i in range(100)]

def draw_bounding_box(frame, bbox, label=None,color=None):
    top,right,bottom,left = bbox

    if color is None: # default color
        color = (0,0,255)
    # import pdb;pdb.set_trace()
    cv2.rectangle(frame,(left,top),(right,bottom),color,2)
    if label is not None:
        cv2.rectangle(frame, (left, bottom), (right, bottom+15), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 3, bottom + 10), font, 0.5, (255, 255, 255), 1)
