# src/avenue/eval_avenue.py
import os
import glob
import csv
import cv2
import numpy as np
import torch
import scipy.io as sio
from src.avenue.train_avenue_cae import Conv3dAE

# ======== MUST MATCH TRAINING ========
WIN = 5
H, W = 112, 160                      # model's grayscale input size (from prepare_avenue)
MODEL_PATH = "models/avenue_c3d.pt"
ROOT = "data/Avenue"                 # dataset root
# =====================================

# ---- Visualization & postproc knobs ----
SMOOTH_KSIZE = 7          # Gaussian blur kernel
MORPH_K = 3               # morphology kernel size
MIN_AREA = 120            # drop tiny blobs (px)
OVERLAY_ALPHA = 0.0    # 0..1 (weight of heatmap)
DRAW_MODE = "boxes_only"     # "overlay" | "boxes_only" | "side-by-side"
WRITE_VIDEO = True        # set False to only produce CSV (faster I/O)

# Keys we’ll try to find in the .mat files
LIKELY_KEYS = [
    # cell arrays (boxes)
    "gt", "gtFrames", "groundtruth", "anno", "bboxes", "boxes", "BB",
    # mask volumes
    "mask", "masks", "volLabel", "LABELS", "gtMask", "vol"
]

# ---------- GT loading & helpers ----------
def _mask_to_boxes(mask2d):
    m = (mask2d.astype(np.uint8) > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_AREA:
            boxes.append([float(x), float(y), float(w), float(h)])
    return boxes

def load_gt_rects(mat_path):
    """Return list[frame] -> list[[x,y,w,h], ...]; resizes masks to (H,W)."""
    m = sio.loadmat(mat_path)
    key = None
    for k in LIKELY_KEYS:
        if k in m:
            key = k
            break
    if key is None:
        for k in m.keys():
            if not k.startswith("__"):
                key = k
                break
    if key is None:
        raise KeyError(f"No usable key in {mat_path}")

    arr = m[key]
    # one-time debug
    if os.environ.get("EVAL_DEBUG_ONCE") != "1":
        print(f"[GT] {os.path.basename(mat_path)} -> key='{key}', shape={getattr(arr,'shape',None)}")
        os.environ["EVAL_DEBUG_ONCE"] = "1"

    # A) cell array of boxes
    if isinstance(arr, np.ndarray) and arr.dtype == "O":
        rects = []
        for cell in arr.ravel():
            if isinstance(cell, np.ndarray) and cell.size > 0:
                a = np.array(cell, dtype=np.float32).reshape(-1, 4)
                rects.append(a.tolist())
            else:
                rects.append([])
        return rects

    # B) 3D mask volume HxWxF (e.g., 'vol')
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] >= 1:
        Hm, Wm, F = arr.shape
        rects = []
        for f in range(F):
            mask = arr[:, :, f]
            if (Hm, Wm) != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            rects.append(_mask_to_boxes(mask))
        return rects

    # C) 1D object array of 2D masks
    if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype == "O":
        rects = []
        for mask in arr:
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                rects.append(_mask_to_boxes(mask))
            else:
                rects.append([])
        return rects

    # D) Single Nx4
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 4:
        return [arr.astype(np.float32).tolist()]

    raise KeyError(f"Unrecognized GT structure for key '{key}' with shape {getattr(arr,'shape',None)} in {mat_path}")

# ---------- frame readers ----------
def frame_clip_reader(frames_dir):
    """Small grayscale frames for the model."""
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    imgs = []
    for f in frames:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.resize(g, (W, H)).astype(np.float32) / 255.0
        imgs.append(g)
    return np.stack(imgs, axis=0) if imgs else np.zeros((0, H, W), np.float32)

def color_frame_reader(frames_dir_rgb):
    """Full-resolution COLOR frames for visualization."""
    frames = sorted(glob.glob(os.path.join(frames_dir_rgb, "*.jpg")))
    imgs = []
    for f in frames:
        img = cv2.imread(f, cv2.IMREAD_COLOR)  # BGR
        if img is not None:
            imgs.append(img)
    return imgs

def windowize(frames, win=WIN):
    X, F = [], frames.shape[0]
    for i in range(F - win + 1):
        X.append(frames[i:i + win][None, ...])  # (1,T,H,W)
    return np.stack(X, axis=0) if X else np.zeros((0, 1, win, H, W), np.float32)

# ---------- error post-processing ----------
def postprocess_err(err, th_global):
    # crop a thin border (reduce edge flicker)
    err = err.copy()
    err[:2, :] = 0; err[-2:, :] = 0; err[:, :2] = 0; err[:, -2:] = 0
    # smooth
    err_s = cv2.GaussianBlur(err, (SMOOTH_KSIZE, SMOOTH_KSIZE), 0)
    # adaptive threshold (mean + 2*std) + global
    mu, sig = float(err_s.mean()), float(err_s.std() + 1e-6)
    th_adapt = mu + 2.0 * sig
    th = max(th_global, th_adapt)
    # binary + morphology
    m = (err_s >= th).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    # contours -> boxes
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_AREA:
            boxes.append([x, y, w, h])
    return err_s, th, boxes

def iou(a, b):
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = aw * ah + bw * bh - inter + 1e-6
    return inter / ua

# ---------- main ----------
def main():
    # discover clips & mats (TEST)
    test_dirs = sorted(glob.glob(os.path.join(ROOT, "frames_test", "*")))
    gt_mats   = sorted(glob.glob(os.path.join(ROOT, "annotations", "test", "*.mat")))
    if len(gt_mats) != len(test_dirs):
        print(f"[WARN] test clips={len(test_dirs)} vs GT mats={len(gt_mats)} — continuing with min length.")
    N = min(len(test_dirs), len(gt_mats))
    test_dirs, gt_mats = test_dirs[:N], gt_mats[:N]

    # model
    model = Conv3dAE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # pass 1: global threshold from test scores (95th percentile)
    all_scores = []
    for clip_dir in test_dirs:
        frames_gray = frame_clip_reader(clip_dir)
        X = windowize(frames_gray)
        for i in range(X.shape[0]):
            x = torch.from_numpy(X[i:i + 1])
            with torch.no_grad():
                recon = model(x).numpy()[0, 0]
            orig_small = X[i, 0]
            c = WIN // 2
            err = (orig_small[c] - recon[c]) ** 2
            all_scores.append(float(err.mean()))
    if not all_scores:
        print("No test windows found."); return
    th_frame = float(np.percentile(all_scores, 95.0))
    th_pixel = th_frame * 2.0
    print(f"Thresholds → frame={th_frame:.6f}  pixel={th_pixel:.6f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/avenue_alerts.csv", "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv); wcsv.writerow(["clip", "frame_idx", "mean_error", "alert"])
        iou_hits, iou_total = 0, 0

        for clip_dir, gt_mat in zip(test_dirs, gt_mats):
            clip_name = os.path.basename(clip_dir)

            # small gray frames for model
            frames_gray = frame_clip_reader(clip_dir)
            X = windowize(frames_gray)
            gt = load_gt_rects(gt_mat)

            # COLOR frames for visualization (native res)
            clip_dir_rgb = clip_dir.replace("frames_test", "frames_test_rgb")
            frames_rgb = color_frame_reader(clip_dir_rgb)
            Hc, Wc = (frames_rgb[0].shape[:2] if frames_rgb else (H, W))

            if WRITE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join("outputs", f"{clip_name}_annot.mp4")
                vw = cv2.VideoWriter(out_path, fourcc, 25, (Wc, Hc))

            for i in range(X.shape[0]):
                x = torch.from_numpy(X[i:i + 1])
                with torch.no_grad():
                    recon = model(x).numpy()[0, 0]
                orig_small = X[i, 0]
                c = WIN // 2
                err = (orig_small[c] - recon[c]) ** 2
                score = float(err.mean())
                is_alert = score >= th_frame
                frame_idx = i + c
                wcsv.writerow([clip_name, frame_idx, f"{score:.6f}", "anomaly" if is_alert else ""])

                # denoise + boxes on small map
                err_s, th_used, det_boxes_small = postprocess_err(err, th_pixel)

                # upscale boxes & (optionally) heatmap to color resolution
                scale_x = Wc / float(W)
                scale_y = Hc / float(H)
                det_boxes = [[int(x0 * scale_x), int(y0 * scale_y),
                              int(w0 * scale_x), int(h0 * scale_y)]
                             for x0, y0, w0, h0 in det_boxes_small]

                # current GT rects for this frame (also upscale)
                gt_rects_small = gt[frame_idx] if frame_idx < len(gt) else []
                gt_boxes_big = [[int(xg * scale_x), int(yg * scale_y),
                                 int(wg * scale_x), int(hg * scale_y)]
                                for xg, yg, wg, hg in gt_rects_small]

                # IoU metric
                for g in gt_boxes_big:
                    best = 0.0
                    for d in det_boxes:
                        best = max(best, iou(d, g))
                    iou_hits += (best >= 0.2)
                    iou_total += 1

                if WRITE_VIDEO:
                    if frame_idx < len(frames_rgb):
                        base_bgr = frames_rgb[frame_idx].copy()
                    else:
                        base = (orig_small[c] * 255.0).clip(0, 255).astype(np.uint8)
                        base_bgr = cv2.cvtColor(cv2.resize(base, (Wc, Hc), interpolation=cv2.INTER_CUBIC),
                                                cv2.COLOR_GRAY2BGR)

                    if DRAW_MODE in ("overlay", "side-by-side"):
                        norm = (np.clip(err_s / (th_used * 2.0), 0, 1) * 255).astype(np.uint8)
                        heat_small = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                        heat_big = cv2.resize(heat_small, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
                        overlay = cv2.addWeighted(base_bgr, 1.0 - OVERLAY_ALPHA, heat_big, OVERLAY_ALPHA, 0)
                        vis = overlay if DRAW_MODE == "overlay" else np.hstack([base_bgr, heat_big])
                    else:  # boxes_only
                        vis = base_bgr

                    # draw GT (green)
                    for xg, yg, wg, hg in gt_boxes_big:
                        cv2.rectangle(vis, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

                    # draw detections (red)
                    for xd, yd, wd, hd in det_boxes:
                        cv2.rectangle(vis, (xd, yd), (xd + wd, yd + hd), (0, 0, 255), 2)

                    # optional unsharp mask (a touch sharper)
                    sharp = cv2.GaussianBlur(vis, (0, 0), 1.0)
                    vis = cv2.addWeighted(vis, 1.5, sharp, -0.5, 0)

                    txt = f"{clip_name} f={frame_idx}  score={score:.5f}  th={th_used:.5f}  {'ANOM' if is_alert else 'ok'}"
                    cv2.putText(vis, txt, (8, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    vw.write(vis)

            if WRITE_VIDEO:
                vw.release()

        if iou_total > 0:
            print(f"Localization hit-rate (IoU>=0.2): {iou_hits / iou_total:.3f}")
        print("Wrote: outputs/avenue_alerts.csv", end="")
        if WRITE_VIDEO:
            print(" and outputs/<clip>_annot.mp4")
        else:
            print("")

if __name__ == "__main__":
    main()
