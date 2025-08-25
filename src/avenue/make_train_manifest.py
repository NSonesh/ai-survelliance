import os, glob, csv, cv2
import numpy as np
import scipy.io as sio

H, W = 112, 160   # MUST match your training resize

def mask_to_bool(mask3d):
    Hm, Wm, F = mask3d.shape
    bad = []
    for f in range(F):
        m = mask3d[:,:,f]
        if (Hm, Wm) != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        bad.append(bool(np.any(m)))
    return np.array(bad, dtype=bool)

def find_key(m):
    for k in ("vol","mask","masks","volLabel","LABELS","gtMask","gt","gtFrames","groundtruth","anno"):
        if k in m: return k
    for k in m.keys():
        if not k.startswith("__"): return k
    raise KeyError("No usable key in mat")

def main():
    frames_root = "data/Avenue/frames_train"
    mats_root   = "data/Avenue/annotations/train"

    clips = sorted(glob.glob(os.path.join(frames_root, "*")))
    mats  = sorted(glob.glob(os.path.join(mats_root, "*.mat")))
    assert len(clips) == len(mats), f"Mismatch train clips={len(clips)} vs mats={len(mats)}"

    os.makedirs("manifests", exist_ok=True)
    out_csv = "manifests/avenue_train_clean_frames.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip_dir","frame_idx","is_clean"])
        for clip_dir, mat_path in zip(clips, mats):
            m = sio.loadmat(mat_path); key = find_key(m); arr = m[key]
            frames = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
            F = len(frames)
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                bad = mask_to_bool(arr)             # True where anomaly mask exists
            else:
                bad = np.zeros((F,), dtype=bool)    # if unexpected format, assume all clean
            n = min(F, bad.shape[0])
            for i in range(n):   w.writerow([clip_dir, i, 0 if not bad[i] else 1])
            for i in range(n,F): w.writerow([clip_dir, i, 0])
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
