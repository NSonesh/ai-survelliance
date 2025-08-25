# src/avenue/train_avenue_cae.py
# 3D-Conv Autoencoder for Avenue (fast-friendly) + training manifest to skip dirty frames
import os, glob, cv2, random, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------ perf helpers ------------------------
def set_cpu_fast(num_threads=4):
    """Make CPU training snappier (Windows-friendly)."""
    torch.set_num_threads(num_threads)
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

# ------------------------ manifest utils ------------------------
def load_clean_map(csv_path):
    """
    Reads manifests/avenue_train_clean_frames.csv created by make_train_manifest.py
    Returns: dict clip_dir -> set(bad_frame_idxs)
    where "bad" means frame contains anomaly and should be excluded from training windows.
    CSV columns: clip_dir,frame_idx,is_clean(0/1)  [0 = clean, 1 = not clean]
    """
    import csv
    bad = {}
    if not csv_path or not os.path.exists(csv_path):
        return bad
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            cdir = row["clip_dir"]
            i = int(row["frame_idx"])
            is_clean = int(row["is_clean"]) == 0
            if not is_clean:
                bad.setdefault(cdir, set()).add(i)
    return bad

# ------------------------ dataset ------------------------
class ClipsDataset(Dataset):
    """
    Builds T-frame windows from pre-extracted Avenue frames.
    Skips any window that overlaps a frame marked "dirty" by the training manifest.
    """
    def __init__(self, frames_root, win=5, size=(160,120), frame_stride=2,
                 max_clips=None, limit_per_clip=None, train_manifest=None):
        self.win = win
        self.W, self.H = size
        self.frame_stride = max(1, frame_stride)

        clip_dirs = sorted(glob.glob(os.path.join(frames_root, "*")))
        if max_clips:
            clip_dirs = clip_dirs[:max_clips]

        # map of clip_dir -> set(bad_frames)
        bad_map = load_clean_map(train_manifest)

        self.index = []  # list of (clip_dir, start_index)
        for cdir in clip_dirs:
            frames = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
            if len(frames) < win:
                continue
            starts = list(range(0, len(frames) - win + 1, self.frame_stride))
            if limit_per_clip:
                starts = starts[:limit_per_clip]

            bad_set = bad_map.get(cdir, set())
            for s in starts:
                # skip if any frame in the window is bad
                if any((s + k) in bad_set for k in range(win)):
                    continue
                self.index.append((cdir, s))

        random.shuffle(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cdir, s = self.index[idx]
        xs = []
        # load a T-frame window
        for k in range(s, s + self.win):
            fp = os.path.join(cdir, f"{k:06d}.jpg")
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # robust fallback: find the k-th frame by glob if named differently
                files = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
                img = cv2.imread(files[k], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.W, self.H)).astype(np.float32) / 255.0
            xs.append(img)
        x = np.stack(xs, axis=0)  # (T,H,W)
        x = x[None, ...]          # (C=1,T,H,W)
        return torch.from_numpy(x)

# ------------------------ model ------------------------
class Conv3dAE(nn.Module):
    """
    Very compact 3D-Conv autoencoder:
      in:  (B,1,T,H,W)
      out: (B,1,T,H,W)
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, (3,3,3), stride=(1,2,2), padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16,32,(3,3,3), stride=(1,2,2), padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32,64,(3,3,3), stride=(1,2,2), padding=1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(64,32,(3,4,4), stride=(1,2,2), padding=(1,1,1)), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32,16,(3,4,4), stride=(1,2,2), padding=(1,1,1)), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1,(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.dec(self.enc(x))

# ------------------------ train loop ------------------------
def train(frames_root, epochs, lr, bs, device, win, size, frame_stride,
          max_clips, limit_per_clip, num_workers, train_manifest):
    ds = ClipsDataset(frames_root, win=win, size=size, frame_stride=frame_stride,
                      max_clips=max_clips, limit_per_clip=limit_per_clip,
                      train_manifest=train_manifest)
    if len(ds) == 0:
        raise RuntimeError("No training windows found. "
                           "Check frames_root, WIN/size, and that the manifest didn't filter everything out.")
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)

    model = Conv3dAE().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    os.makedirs("models", exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x in dl:
            x = x.to(device, non_blocking=True)
            y = model(x)
            loss = crit(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)

        print(f"Epoch {ep}/{epochs} - MSE: {tot/len(ds):.6f} (windows: {len(ds)})")
        torch.save(model.state_dict(), "models/avenue_c3d.pt")
    print("Saved models/avenue_c3d.pt")

# ------------------------ CLI ------------------------
if __name__ == "__main__":
    set_cpu_fast()
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="data/Avenue/frames_train",
                    help="Pre-extracted Avenue training frames root")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Fewer for quick CPU tests; raise for quality (e.g., 30-80)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--win", type=int, default=5, help="Temporal window length")
    ap.add_argument("--size", type=str, default="112x160",
                    help="HxW (must match prepare_avenue + evaluator). e.g., 112x160")
    ap.add_argument("--frame_stride", type=int, default=6,
                    help="Larger stride -> fewer windows -> faster")
    ap.add_argument("--max_clips", type=int, default=None,
                    help="Limit number of clips for quick runs (e.g., 6)")
    ap.add_argument("--limit_per_clip", type=int, default=None,
                    help="Limit windows per clip for quick runs (e.g., 300)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Windows: keep 0 to avoid spawn overhead")
    ap.add_argument("--train_manifest", type=str, default="manifests/avenue_train_clean_frames.csv",
                    help="CSV from make_train_manifest.py (skips dirty frames). Set empty to disable.")
    args = ap.parse_args()

    H = int(args.size.split("x")[0]); W = int(args.size.split("x")[1])
    train(args.frames_root, args.epochs, args.lr, args.bs, args.device,
          args.win, (W, H), args.frame_stride, args.max_clips, args.limit_per_clip,
          args.num_workers, args.train_manifest)
