# src/avenue/prepare_avenue.py
import os
import cv2
import glob

# Small grayscale size used by the model (W, H)
SIZE_GRAY = (160, 112)   # => H=112, W=160 in the trainer/evaluator

def extract(video_path, out_gray_dir, out_color_dir):
    os.makedirs(out_gray_dir, exist_ok=True)
    os.makedirs(out_color_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Save COLOR frame at native resolution for visualization
        cv2.imwrite(os.path.join(out_color_dir, f"{i:06d}.jpg"), frame)

        # 2) Save GRAYSCALE, resized for the model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, SIZE_GRAY)
        cv2.imwrite(os.path.join(out_gray_dir, f"{i:06d}.jpg"), gray)

        i += 1
    cap.release()
    return i

def main():
    root = "data/Avenue"
    trains = sorted(glob.glob(os.path.join(root, "training_videos", "*.*")))
    tests  = sorted(glob.glob(os.path.join(root, "testing_videos", "*.*")))

    # Training clips
    for v in trains:
        name = os.path.splitext(os.path.basename(v))[0]
        n = extract(
            v,
            os.path.join(root, "frames_train", name),       # gray, small
            os.path.join(root, "frames_train_rgb", name)    # color, native
        )
        print("Train:", name, n, "frames")

    # Test clips
    for v in tests:
        name = os.path.splitext(os.path.basename(v))[0]
        n = extract(
            v,
            os.path.join(root, "frames_test", name),        # gray, small
            os.path.join(root, "frames_test_rgb", name)     # color, native
        )
        print("Test:", name, n, "frames")

if __name__ == "__main__":
    main()
