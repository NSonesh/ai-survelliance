import time
from src.utils.geometry import l2

class AbandonmentDetector:
    def __init__(self, watch_classes, radius_pixels, seconds_threshold):
        self.watch_classes = set(watch_classes)
        self.radius = radius_pixels
        self.seconds_threshold = seconds_threshold
        self.last_near_person = {}  # obj_track_id -> last_time_near_person

    def update(self, frame, tracks, fps):
        now = time.time()
        persons = [tr for tr in tracks.values() if tr.cls_name == 'person']
        objects = [tr for tr in tracks.values() if tr.cls_name in self.watch_classes]
        alerts = []
        for obj in objects:
            near = False
            for p in persons:
                if l2(p.centroid, obj.centroid) <= self.radius:
                    near = True; break
            if near:
                self.last_near_person[obj.id] = now
            else:
                last = self.last_near_person.get(obj.id, now)
                idle = now - last
                if idle >= self.seconds_threshold:
                    alerts.append(('abandonment', obj.id, {'idle_seconds': round(idle,1)}))
        return alerts
