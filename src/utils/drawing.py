import cv2
import time

def draw_box(frame, box, color=(0,255,0), label=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def draw_roi(frame, roi, color=(255,255,0)):
    x1,y1,x2,y2 = map(int, roi)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, "ROI", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def banner(frame, text, color=(0,0,0), bg=(0,255,255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w, 28), bg, -1)
    cv2.putText(frame, text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def put_ts(frame, ts_struct):
    import time as _t
    if isinstance(ts_struct, (int,float)):
        ts_struct = _t.localtime(ts_struct)
    cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S", ts_struct), (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
