import cv2, numpy as np

YOLO_CLASSES_PERSON = {'person'}
YOLO_CLASSES_OBJECTS = {'backpack','handbag','suitcase','book','sports ball','box','package'}

class Detector:
    def __init__(self, model_name='yolov5s', conf_thres=0.35):
        self.model_name = model_name
        self.conf_thres = conf_thres
        self.model = None
        self.using_yolo = False
        self._try_load_yolo()
        if not self.using_yolo:
            self._init_fallback()

    def _try_load_yolo(self):
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)
            self.model.conf = self.conf_thres
            self.using_yolo = True
        except Exception:
            self.model = None
            self.using_yolo = False

    def _init_fallback(self):
        # HOG person detector as fallback; objects in synthetic videos are red boxes we can detect by color
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """Returns list of dicts: {box:(x1,y1,x2,y2), conf:float, cls_name:str}"""
        out = []
        if self.using_yolo:
            res = self.model(frame, size=640)
            for *xyxy, conf, cls in res.xyxy[0].tolist():
                cls = int(cls)
                name = self.model.names[cls]
                out.append({'box': tuple(map(float, xyxy)), 'conf': float(conf), 'cls_name': name})
            return out
        # fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
        for (x,y,w,h), conf in zip(rects, weights):
            if conf > 0.5:
                out.append({'box': (float(x),float(y),float(x+w),float(y+h)), 'conf': float(conf), 'cls_name':'person'})
        # detect red-ish boxes (synthetic packages)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0,120,70]); upper1 = np.array([10,255,255])
        lower2 = np.array([170,120,70]); upper2 = np.array([180,255,255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 400:
                out.append({'box': (float(x),float(y),float(x+w),float(y+h)), 'conf': 0.8, 'cls_name':'package'})
        return out
