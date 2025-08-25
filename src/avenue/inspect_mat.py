# src/avenue/inspect_mat.py
import sys, pprint
import scipy.io as sio
import numpy as np

def main(path):
    m = sio.loadmat(path)
    print("=== Keys ===")
    print([k for k in m.keys() if not k.startswith("__")])

    for k,v in m.items():
        if k.startswith("__"): 
            continue
        if isinstance(v, np.ndarray):
            print(f"\nKey: {k}  type: ndarray  dtype: {v.dtype}  shape: {v.shape}")
            # show a little more if this looks like per-frame cell/array
            if v.dtype == "O":  # cell array
                print(f"  cell len: {len(v.ravel())}")
                # show shapes of first few cells
                for i,cell in enumerate(v.ravel()[:3]):
                    arr = cell
                    if isinstance(arr, np.ndarray):
                        print(f"   cell[{i}] shape: {arr.shape} dtype: {arr.dtype}")
            else:
                # maybe 3D mask HxWxF
                if v.ndim == 3:
                    H,W,F = v.shape
                    print(f"  looks like 3D array (maybe masks?) H={H} W={W} F={F}")
        else:
            print(f"\nKey: {k}  type: {type(v)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.avenue.inspect_mat <path_to_mat>")
        sys.exit(1)
    main(sys.argv[1])
