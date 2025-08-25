import cv2, numpy as np, os, random

def clip(val, lo, hi): 
    return max(lo, min(hi, val))

def gen_demo(path='data/samples/demo_synth_abandon.mp4', w=960, h=540, fps=30, seconds=25):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(path, fourcc, fps, (w,h))
    # person: blue rectangle; package: red rectangle
    px, py = 100, h-160
    vx, vy = 4, 0
    package = None
    loiter_roi = (int(0.1*w), int(0.6*h), int(0.9*w), int(0.95*h))
    for f in range(int(fps*seconds)):
        frame = np.full((h,w,3), 40, np.uint8)
        # draw ground
        cv2.rectangle(frame, (0,int(0.6*h)), (w,h), (60,60,60), -1)
        # draw ROI
        cv2.rectangle(frame, (loiter_roi[0], loiter_roi[1]), (loiter_roi[2], loiter_roi[3]), (255,255,0), 2)
        # move person
        px += vx; py += vy
        if px > w-80: vx = -abs(vx)
        if px < 20: vx = abs(vx)
        # loiter: slow down inside ROI for a while
        if loiter_roi[0] < px < loiter_roi[2] and random.random() < 0.3:
            vx = clip(vx + random.choice([-1,0,1]), -2, 2)
        # sudden sprint to trigger speed anomaly
        if f == int(fps*16):
            vx = 12
        # draw person
        cv2.rectangle(frame, (int(px), int(py)), (int(px+40), int(py+120)), (200,150,0), -1)  # body (blue-ish)
        # drop package at ~8s
        if f == int(fps*8):
            package = (int(px+20), int(py+110), 60, 40)  # x,y,w,h
        # draw / persist package
        if package:
            x,y,w0,h0 = package
            cv2.rectangle(frame, (x,y), (x+w0,y+h0), (0,0,255), -1)  # red
        vw.write(frame)
    vw.release()
    return path

if __name__ == '__main__':
    p = gen_demo()
    print('Wrote', p)
