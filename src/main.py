import argparse
import cv2, yaml, time, os
from src.utils.yolo import Detector, YOLO_CLASSES_PERSON, YOLO_CLASSES_OBJECTS
from src.utils.drawing import draw_box, draw_roi, banner, put_ts
from src.utils.io import AlertLogger
from src.tracking.centroid_tracker import CentroidTracker
from src.detectors.loitering import LoiteringDetector
from src.detectors.abandonment import AbandonmentDetector
from src.detectors.speed_anomaly import SpeedAnomalyDetector



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=str, default='data/samples/demo_synth_abandon.mp4', help='Path to input video')
    ap.add_argument('--config', type=str, default='src/config.yaml')
    ap.add_argument('--show', action='store_true', help='Display windows')
    ap.add_argument('--save', action='store_true', help='Save annotated video per config')
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config,'r') as f:
        cfg = yaml.safe_load(f)
    det_conf = cfg['detector']['conf_threshold']
    model_name = cfg['detector']['model']
    det = Detector(model_name=model_name, conf_thres=det_conf)
    tracker = CentroidTracker(max_disappeared=cfg['tracking']['max_disappeared'],
                              max_distance=cfg['tracking']['max_distance'])
    loit = LoiteringDetector(cfg['loitering']['roi'], cfg['loitering']['seconds_threshold']) if cfg['loitering']['enabled'] else None
    aban = AbandonmentDetector(cfg['abandonment']['watch_classes'], cfg['abandonment']['radius_pixels'], cfg['abandonment']['seconds_threshold']) if cfg['abandonment']['enabled'] else None
    speed = SpeedAnomalyDetector(cfg['speed_anomaly']['baseline_fps_guess'], cfg['speed_anomaly']['px_per_meter_guess'], cfg['speed_anomaly']['speed_threshold_mps']) if cfg['speed_anomaly']['enabled'] else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(cfg['output']['annotated_video']), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(cfg['output']['annotated_video'], fourcc, fps, (w,h))
    logger = AlertLogger(cfg['output']['alerts_csv'])

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_i += 1
        dets = det.detect(frame)
        keep = [d for d in dets if d['cls_name'] in YOLO_CLASSES_PERSON or d['cls_name'] in YOLO_CLASSES_OBJECTS]
        tracks = tracker.update(keep)
        for tid, tr in tracks.items():
            color = (0,255,0) if tr.cls_name=='person' else (0,140,255)
            draw_box(frame, tr.box, color, f'{tr.cls_name} #{tid}')
        alerts = []
        if loit:
            a, roi = loit.update(frame, tracks, fps)
            alerts += a
            draw_roi(frame, roi)
        if aban: alerts += aban.update(frame, tracks, fps)
        if speed: alerts += speed.update(frame, tracks, fps)
        if alerts:
            for typ, tid, details in alerts:
                logger.log(int(time.time()), typ, tid, details)
            banner(frame, 'ALERTS: ' + ' | '.join([f"{t}:{tid}:{d}" for t,tid,d in alerts]))
        put_ts(frame, time.time())
        if args.show:
            cv2.imshow('annotated', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if writer: writer.write(frame)
    logger.close()
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
