import os, cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
from glob import glob
from tqdm import tqdm

class SmallCAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid(),
        )
    def forward(self,x):
        z = self.enc(x)
        x2 = self.dec(z)
        return x2

def load_frames_from_dir(d, size=(160,120), limit=None):
    files = sorted(glob(os.path.join(d,'*.jpg')) + glob(os.path.join(d,'*.png')))
    if limit: files = files[:limit]
    xs = []
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, size)
        xs.append(img.astype(np.float32)/255.0)
    return np.stack(xs) if xs else np.zeros((0,size[1],size[0]), np.float32)

def train(data_dir='data/normal_frames', epochs=5, lr=1e-3, bs=32, device='cpu'):
    os.makedirs('models', exist_ok=True)
    X = load_frames_from_dir(data_dir, limit=None)
    if len(X)==0:
        print('No frames at', data_dir); return
    X = torch.from_numpy(X[:,None,:,:])
    model = SmallCAE().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(1,epochs+1):
        perm = torch.randperm(X.size(0))
        tot = 0.0
        for i in range(0, X.size(0), bs):
            idx = perm[i:i+bs]
            xb = X[idx].to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f'Epoch {ep}: MSE={tot/X.size(0):.6f}')
    torch.save(model.state_dict(), 'models/cae_small.pt')
    print('Saved models/cae_small.pt')

if __name__ == '__main__':
    train()
