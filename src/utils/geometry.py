import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = boxAArea + boxBArea - interArea + 1e-6
    return interArea / union

def rect_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def l2(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def denorm_roi(roi_norm, w, h):
    x1 = int(roi_norm[0] * w)
    y1 = int(roi_norm[1] * h)
    x2 = int(roi_norm[2] * w)
    y2 = int(roi_norm[3] * h)
    return (x1,y1,x2,y2)

def point_in_rect(pt, rect):
    x,y = pt
    x1,y1,x2,y2 = rect
    return (x>=x1 and x<=x2 and y>=y1 and y<=y2)
