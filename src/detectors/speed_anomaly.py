class SpeedAnomalyDetector:
    def __init__(self, baseline_fps_guess=30, px_per_meter_guess=50, speed_threshold_mps=4.5):
        self.fps_guess = baseline_fps_guess
        self.ppm = px_per_meter_guess
        self.thresh = speed_threshold_mps

    def update(self, frame, tracks, fps):
        alerts = []
        fps_used = fps if fps and fps > 0 else self.fps_guess
        for tid, tr in tracks.items():
            if tr.cls_name != 'person': 
                continue
            mps = (tr.avg_speed * fps_used) / max(1e-6, self.ppm)
            if mps >= self.thresh:
                alerts.append(('speed_anomaly', tid, {'speed_mps': round(mps,2)}))
        return alerts
