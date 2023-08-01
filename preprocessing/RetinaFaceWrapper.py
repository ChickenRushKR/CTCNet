import cv2
import numpy as np
from retinaface import RetinaFace

def detect_face(face_detector, img, isExclude448 = True, scale = 1.05, tran = 0.):


    regins = []

    obj = face_detector.predict(rgb_image = img, threshold=0.9)
    tag  = False
    for identity in obj:

        x = identity['x1']
        w = identity['x2'] - x
        y = identity['y1']
        h = identity['y2'] - y

        # old_size = (h + w) / 2
        old_size = max(h, w)
        center = np.array([identity['x2'] - w / 2.0, identity['y2'] - h / 2.0+old_size*tran])


        # old_size = (h + w) / 2
        # center = np.array([identity['y2'] - h / 2.0, identity['x2'] - w / 2.0 +old_size*0.12])

        size = int(old_size * scale)
        if isExclude448:
            if size < 448:
                tag = True
                continue
        img_region = [center[0] - size / 2,center[1] - size / 2,center[0] + size / 2, center[1] + size / 2, 1]
        regins.append(img_region)

    return regins, tag
