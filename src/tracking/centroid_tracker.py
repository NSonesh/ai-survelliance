from collections import deque
from src.utils.geometry import rect_center, l2

class Track:
    def __init__(self, track_id, box, cls_name):
        self.id = track_id
        self.box = box
        self.cls_name = cls_name
        self.centroid = rect_center(box)
        self.history = deque(maxlen=30)
        self.history.append(self.centroid)
        self.frames_since_seen = 0
        self.age = 0
        self.avg_speed = 0.0  # pixels/frame

    def update(self, box):
        self.box = box
        c = rect_center(box)
        self.history.append(c)
        if len(self.history) >= 2:
            d = l2(self.history[-2], self.history[-1])
            self.avg_speed = 0.8*self.avg_speed + 0.2*d
        self.centroid = c
        self.frames_since_seen = 0
        self.age += 1

class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        self.next_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        # detections: list of dicts {'box':(x1,y1,x2,y2), 'cls_name':str, 'conf':float}
        used = set()
        det_centers = [rect_center(d['box']) for d in detections]
        det_classes = [d['cls_name'] for d in detections]

        # match existing tracks
        for tid, tr in list(self.tracks.items()):
            best_i = -1; best_d = 1e9
            for i,(c,cls) in enumerate(zip(det_centers, det_classes)):
                if i in used: 
                    continue
                if cls != tr.cls_name: 
                    continue
                d = ((tr.centroid[0]-c[0])**2 + (tr.centroid[1]-c[1])**2) ** 0.5
                if d < best_d:
                    best_d = d; best_i = i
            if best_i != -1 and best_d <= self.max_distance:
                tr.update(detections[best_i]['box'])
                used.add(best_i)
            else:
                tr.frames_since_seen += 1
                tr.age += 1
                if tr.frames_since_seen > self.max_disappeared:
                    del self.tracks[tid]

        # new tracks
        for i,det in enumerate(detections):
            if i in used: 
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid, det['box'], det['cls_name'])
        return self.tracks
