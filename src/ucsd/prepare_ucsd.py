# src/ucsd/prepare_ucsd.py
import os, glob, cv2

# model input size (W,H) -> trainer/evaluator use H=112, W=160
SIZE_GRAY = (160, 112)

def extract(video_path, out_gray_dir, out_color_dir):
    os.makedirs(out_gray_dir, exist_ok=True)
    os.makedirs(out_color_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        # color (native)
        cv2.imwrite(os.path.join(out_color_dir, f"{i:06d}.jpg"), frame)
        # gray (resized)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, SIZE_GRAY)
        cv2.imwrite(os.path.join(out_gray_dir, f"{i:06d}.jpg"), g)
        i += 1
    cap.release()
    return i

def prep_split(root, split="Train"):
    vids = sorted(glob.glob(os.path.join(root, split, "*.*")))
    for v in vids:
        name = os.path.splitext(os.path.basename(v))[0]
        n = extract(
            v,
            os.path.join(root, f"frames_{split.lower()}", name),
            os.path.join(root, f"frames_{split.lower()}_rgb", name),
        )
        print(f"{os.path.basename(root)} {split}: {name} {n} frames")

def main():
    # expected layout:
    # data/UCSD/Ped1/Train/*.avi, Test/*.avi
    # data/UCSD/Ped2/Train/*.avi, Test/*.avi
    for ped in ("Ped1", "Ped2"):
        base = os.path.join("data", "UCSD", ped)
        if os.path.isdir(os.path.join(base, "Train")):
            prep_split(base, "Train")
        if os.path.isdir(os.path.join(base, "Test")):
            prep_split(base, "Test")

if __name__ == "__main__":
    main()
