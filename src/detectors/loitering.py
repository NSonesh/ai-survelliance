import time
from src.utils.geometry import denorm_roi, point_in_rect


class LoiteringDetector:
    def __init__(self, roi_norm, seconds_threshold):
        self.roi_norm = roi_norm
        self.seconds_threshold = seconds_threshold
        self.enter_times = {}  # track_id -> time.time()

    def update(self, frame, tracks, fps):
        h, w = frame.shape[:2]
        roi = denorm_roi(self.roi_norm, w, h)
        alerts = []
        now = time.time()
        for tid, tr in tracks.items():
            if tr.cls_name != 'person': 
                continue
            if point_in_rect(tr.centroid, roi):
                if tid not in self.enter_times:
                    self.enter_times[tid] = now
                dwell = now - self.enter_times[tid]
                if dwell >= self.seconds_threshold:
                    alerts.append(('loitering', tid, {'dwell_seconds': round(dwell,1)}))
            else:
                if tid in self.enter_times:
                    del self.enter_times[tid]
        return alerts, roi
