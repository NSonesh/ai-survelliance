# src/ucsd/eval_ucsd.py
import os, glob, csv, cv2, numpy as np, torch, scipy.io as sio
from src.avenue.train_avenue_cae import Conv3dAE  # reuse the same model class

# ======= MUST MATCH TRAINING =======
WIN = 5
H, W = 112, 160
MODEL_PATH = "models/avenue_c3d.pt"   # or models/ucsd_c3d.pt if you retrain
ROOT = "data/UCSD"                     # dataset root with Ped1, Ped2
# ===================================

# viz / postproc
SMOOTH_KSIZE = 7
MORPH_K = 3
MIN_AREA = 120
OVERLAY_ALPHA = 0.35
DRAW_MODE = "overlay"   # "overlay" | "boxes_only" | "side-by-side"
WRITE_VIDEO = True

# ---------- helpers ----------
def frame_clip_reader(frames_dir):
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    imgs = []
    for f in frames:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        g = cv2.resize(g, (W, H)).astype(np.float32)/255.0
        imgs.append(g)
    return np.stack(imgs, axis=0) if imgs else np.zeros((0,H,W), np.float32)

def color_frame_reader(frames_dir_rgb):
    frames = sorted(glob.glob(os.path.join(frames_dir_rgb, "*.jpg")))
    imgs = []
    for f in frames:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is not None: imgs.append(img)
    return imgs

def windowize(frames, win=WIN):
    X, F = [], frames.shape[0]
    for i in range(F - win + 1):
        X.append(frames[i:i+win][None, ...])
    return np.stack(X, axis=0) if X else np.zeros((0,1,win,H,W), np.float32)

def postprocess_err(err, th_global):
    err = err.copy()
    err[:2,:]=0; err[-2:,:]=0; err[:,:2]=0; err[:,-2:]=0
    err_s = cv2.GaussianBlur(err, (SMOOTH_KSIZE, SMOOTH_KSIZE), 0)
    mu, sig = float(err_s.mean()), float(err_s.std()+1e-6)
    th_adapt = mu + 2.0*sig
    th = max(th_global, th_adapt)
    m = (err_s >= th).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= MIN_AREA:
            boxes.append([x,y,w,h])
    return err_s, th, boxes

def iou(a, b):
    ax1,ay1,aw,ah = a; ax2,ay2=ax1+aw, ay1+ah
    bx1,by1,bw,bh = b; bx2,by2=bx1+bw, by1+bh
    x1,y1=max(ax1,bx1), max(ay1,by1)
    x2,y2=min(ax2,bx2), min(ay2,by2)
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=aw*ah + bw*bh - inter + 1e-6
    return inter/ua

# ---------- UCSD GT loaders ----------
def load_frame_flags(mat_or_txt_path):
    """
    UCSD provides frame-level flags per clip (1=anomaly, 0=normal).
    Supports:
      - .mat with keys like 'frameGt','gt','labels'
      - .txt with 0/1 per line
    Returns list[int] length F (frames); if unavailable, returns [].
    """
    p = mat_or_txt_path
    if not os.path.exists(p): return []
    if p.lower().endswith(".txt"):
        return [int(x.strip()) for x in open(p, "r", encoding="utf-8") if x.strip().isdigit()]
    # .mat
    m = sio.loadmat(p)
    for k in ("frameGt","gt","labels","frame_labels","frame_level_gt"):
        if k in m:
            arr = np.array(m[k]).astype(np.int32).ravel().tolist()
            return arr
    # fallback: first non-__ 1D array
    for k,v in m.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and v.ndim==2 and (v.shape[0]==1 or v.shape[1]==1):
            return v.astype(np.int32).ravel().tolist()
    return []

def load_pixel_masks(mat_path, target_hw=(H,W)):
    """
    Optional pixel masks for a subset of clips (HxWxF). Converts to boxes per frame.
    Returns: list[frame] -> list[[x,y,w,h], ...]
    """
    if not os.path.exists(mat_path): return []
    m = sio.loadmat(mat_path)
    key = None
    for k in ("vol","mask","masks","volLabel","LABELS","gtMask"):
        if k in m: key = k; break
    if key is None:
        for k in m.keys():
            if not k.startswith("__"): key = k; break
    arr = m[key]
    if not (isinstance(arr,np.ndarray) and arr.ndim==3): return []
    Hm,Wm,F = arr.shape
    boxes_per_frame = []
    for f in range(F):
        mask = arr[:,:,f]
        if (Hm,Wm)!=target_hw[::-1]:
            mask = cv2.resize(mask, target_hw, interpolation=cv2.INTER_NEAREST)
        mbin = (mask.astype(np.uint8)>0).astype(np.uint8)*255
        cnts,_ = cv2.findContours(mbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h >= MIN_AREA:
                boxes.append([x,y,w,h])
        boxes_per_frame.append(boxes)
    return boxes_per_frame

# ---------- evaluation per Ped subset ----------
def eval_subset(ped="Ped1"):
    subset_root = os.path.join(ROOT, ped)
    test_dirs = sorted(glob.glob(os.path.join(subset_root, "frames_test", "*")))
    test_dirs_rgb = [d.replace("frames_test", "frames_test_rgb") for d in test_dirs]

    # try to find frame-level flags & mask files; users often place matching names under gt folders
    # expect one GT per clip; customize these globs as per your unpacked UCSD
    frame_flag_files = sorted(glob.glob(os.path.join(subset_root, "Test", "gt", "*.mat")) + 
                              glob.glob(os.path.join(subset_root, "Test", "gt", "*.txt")))
    pixel_mask_files = sorted(glob.glob(os.path.join(subset_root, "Test", "gt_masks", "*.mat")))

    if len(frame_flag_files) != len(test_dirs):
        print(f"[{ped}] WARN: {len(test_dirs)} clips but {len(frame_flag_files)} frame-flag files; using min length.")
    N = min(len(test_dirs), len(frame_flag_files)) if frame_flag_files else len(test_dirs)
    test_dirs, test_dirs_rgb = test_dirs[:N], test_dirs_rgb[:N]
    if frame_flag_files: frame_flag_files = frame_flag_files[:N]

    # model
    model = Conv3dAE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # pass 1: global threshold from test scores (95th percentile)
    all_scores=[]
    for clip_dir in test_dirs:
        frames = frame_clip_reader(clip_dir)
        X = windowize(frames)
        for i in range(X.shape[0]):
            x = torch.from_numpy(X[i:i+1])
            with torch.no_grad():
                recon = model(x).numpy()[0,0]
            orig = X[i,0]
            c = WIN//2
            err = (orig[c]-recon[c])**2
            all_scores.append(float(err.mean()))
    if not all_scores:
        print(f"[{ped}] No test windows found."); return
    th_frame = float(np.percentile(all_scores, 95.0))
    th_pixel = th_frame * 2.0
    print(f"[{ped}] Thresholds → frame={th_frame:.6f}  pixel={th_pixel:.6f}")

    os.makedirs("outputs", exist_ok=True)
    alerts_csv = f"outputs/ucsd_{ped.lower()}_alerts.csv"
    with open(alerts_csv, "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv); wcsv.writerow(["subset","clip","frame_idx","mean_error","pred_alert","gt_flag"])

        # metrics
        tp=fp=tn=fn=0
        iou_hits=iou_total=0

        for idx,(clip_dir,clip_dir_rgb) in enumerate(zip(test_dirs,test_dirs_rgb)):
            clip_name = os.path.basename(clip_dir)
            frames = frame_clip_reader(clip_dir)
            X = windowize(frames)
            flags = load_frame_flags(frame_flag_files[idx]) if frame_flag_files else []
            # optional pixel masks (if present and aligned by index)
            masks_boxes = []
            if pixel_mask_files:
                if idx < len(pixel_mask_files):
                    masks_boxes = load_pixel_masks(pixel_mask_files[idx], target_hw=(H,W))

            # color
            frames_rgb = color_frame_reader(clip_dir_rgb)
            Hc,Wc = (frames_rgb[0].shape[:2] if frames_rgb else (H,W))
            if WRITE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join("outputs", f"ucsd_{ped.lower()}_{clip_name}_annot.mp4")
                vw = cv2.VideoWriter(out_path, fourcc, 25, (Wc, Hc))

            for i in range(X.shape[0]):
                x = torch.from_numpy(X[i:i+1])
                with torch.no_grad():
                    recon = model(x).numpy()[0,0]
                orig = X[i,0]
                c = WIN//2
                err = (orig[c]-recon[c])**2
                score = float(err.mean())
                pred_alert = int(score >= th_frame)
                frame_idx = i + c
                gt_flag = flags[frame_idx] if flags and frame_idx < len(flags) else ""

                wcsv.writerow([ped, clip_name, frame_idx, f"{score:.6f}", pred_alert, gt_flag])

                # postproc -> boxes (small map)
                err_s, th_used, det_boxes_small = postprocess_err(err, th_pixel)
                # up-scale
                sx, sy = Wc/float(W), Hc/float(H)
                det_boxes = [[int(x*sx), int(y*sy), int(w*sx), int(h*sy)] for x,y,w,h in det_boxes_small]

                # frame-level metrics
                if gt_flag in (0,1):
                    if pred_alert==1 and gt_flag==1: tp+=1
                    elif pred_alert==1 and gt_flag==0: fp+=1
                    elif pred_alert==0 and gt_flag==0: tn+=1
                    elif pred_alert==0 and gt_flag==1: fn+=1

                # IoU vs pixel masks if available
                if masks_boxes and frame_idx < len(masks_boxes):
                    gt_boxes_small = masks_boxes[frame_idx]
                    gt_boxes_big = [[int(x*sx), int(y*sy), int(w*sx), int(h*sy)] for x,y,w,h in gt_boxes_small]
                    for g in gt_boxes_big:
                        best = 0.0
                        for d in det_boxes:
                            best = max(best, iou(d,g))
                        iou_hits += (best >= 0.2)
                        iou_total += 1

                if WRITE_VIDEO:
                    base_bgr = frames_rgb[frame_idx].copy() if frame_idx < len(frames_rgb) else \
                               cv2.cvtColor(cv2.resize((orig[c]*255).astype(np.uint8),(Wc,Hc),interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)

                    if DRAW_MODE in ("overlay","side-by-side"):
                        norm = (np.clip(err_s/(max(th_used*2.0,1e-6)),0,1)*255).astype(np.uint8)
                        heat_small = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                        heat_big = cv2.resize(heat_small,(Wc,Hc),interpolation=cv2.INTER_CUBIC)
                        overlay = cv2.addWeighted(base_bgr, 1.0-OVERLAY_ALPHA, heat_big, OVERLAY_ALPHA, 0)
                        vis = overlay if DRAW_MODE=="overlay" else np.hstack([base_bgr, heat_big])
                    else:
                        vis = base_bgr

                    # draw detections
                    for xd,yd,wd,hd in det_boxes:
                        cv2.rectangle(vis,(xd,yd),(xd+wd,yd+hd),(0,0,255),2)
                    # optional GT pixel boxes (green)
                    if masks_boxes and frame_idx < len(masks_boxes):
                        for xg,yg,wg,hg in [[int(x*sx), int(y*sy), int(w*sx), int(h*sy)] for x,y,w,h in masks_boxes[frame_idx]]:
                            cv2.rectangle(vis,(xg,yg),(xg+wg,yg+hg),(0,255,0),2)

                    sharp = cv2.GaussianBlur(vis,(0,0),1.0)
                    vis = cv2.addWeighted(vis,1.5,sharp,-0.5,0)

                    label = "ANOM" if pred_alert else "ok"
                    cv2.putText(vis, f"{ped} {clip_name} f={frame_idx} score={score:.5f} {label}",
                                (8, Hc-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
                    vw.write(vis)

            if WRITE_VIDEO: vw.release()

        # report
        if (tp+tn+fp+fn) > 0:
            acc = (tp+tn)/float(tp+tn+fp+fn)
            prec = tp/float(tp+fp+1e-6); rec = tp/float(tp+fn+1e-6)
            print(f"[{ped}] Frame-level: acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}")
        if iou_total > 0:
            print(f"[{ped}] Localization hit-rate (IoU>=0.2): {iou_hits/iou_total:.3f}")
        print(f"[{ped}] Wrote: {alerts_csv} and outputs/ucsd_{ped.lower()}_*_annot.mp4")

def main():
    for ped in ("Ped1","Ped2"):
        if os.path.isdir(os.path.join(ROOT, ped)):
            eval_subset(ped)

if __name__ == "__main__":
    main()
