"""Entrenamiento de la U-Net de downscaling LST para Arequipa.

Uso:
  python train.py --samples data/samples --epochs 60 --out checkpoints

Split temporal por fecha (evita fuga de informacion): el ultimo 15% de los
dias es validacion y el 15% anterior es test. Reporta RMSE y MBE en grados C
como el paper. Reanudable con --resume.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.unet import UNet, masked_mse          # noqa: E402
from scripts.build_dataset import LST_MEAN, LST_STD  # noqa: E402

CROP = 192


class LSTDataset(Dataset):
    def __init__(self, files, train=True):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x, y, m = d["x"], d["y"], d["mask"].astype(np.float32)
        y = (y - LST_MEAN) / LST_STD
        H, W = y.shape
        if self.train:  # crop aleatorio + flips (augmentacion)
            r = np.random.randint(0, max(1, H - CROP + 1))
            c = np.random.randint(0, max(1, W - CROP + 1))
            x, y, m = x[:, r:r+CROP, c:c+CROP], y[r:r+CROP, c:c+CROP], m[r:r+CROP, c:c+CROP]
            if np.random.rand() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
        else:           # centro fijo, tamano divisible por 8
            H8, W8 = (H // 8) * 8, (W // 8) * 8
            x, y, m = x[:, :H8, :W8], y[:H8, :W8], m[:H8, :W8]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)))


def split_by_date(files):
    dated = sorted(files, key=lambda f: re.search(r"sample_(\d{8})", f)[1])
    dates = sorted({re.search(r"sample_(\d{8})", f)[1] for f in dated})
    n = len(dates)
    test_d = set(dates[int(n * 0.85):])
    val_d = set(dates[int(n * 0.70):int(n * 0.85)])
    tr = [f for f in dated if re.search(r"sample_(\d{8})", f)[1] not in test_d | val_d]
    va = [f for f in dated if re.search(r"sample_(\d{8})", f)[1] in val_d]
    te = [f for f in dated if re.search(r"sample_(\d{8})", f)[1] in test_d]
    return tr, va, te


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    se = err = n = 0.0
    for x, y, m in loader:
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        p = model(x).squeeze(1)
        d = (p - y) * LST_STD          # a Kelvin/gradosC
        se += ((d ** 2) * m).sum().item()
        err += (d * m).sum().item()
        n += m.sum().item()
    return (se / max(n, 1)) ** 0.5, err / max(n, 1)   # RMSE, MBE en C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.samples, "sample_*.npz"))
    assert files, "no hay muestras: corre primero build_dataset.py"
    tr, va, te = split_by_date(files)
    print(f"muestras: {len(tr)} train / {len(va)} val / {len(te)} test")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    ck_last = os.path.join(args.out, "last.pt")
    ck_best = os.path.join(args.out, "best.pt")
    start, best = 0, 1e9
    if args.resume and os.path.exists(ck_last):
        st = torch.load(ck_last, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, best = st["epoch"] + 1, st["best"]
        print(f"reanudando desde epoca {start} (mejor RMSE val {best:.2f} C)")

    dl_tr = DataLoader(LSTDataset(tr, True), batch_size=args.batch,
                       shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(LSTDataset(va, False), batch_size=4, num_workers=2)
    dl_te = DataLoader(LSTDataset(te, False), batch_size=4, num_workers=2)

    for ep in range(start, args.epochs):
        model.train()
        tot = 0.0
        for x, y, m in dl_tr:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            opt.zero_grad()
            loss = masked_mse(model(x), y, m)
            loss.backward()
            opt.step()
            tot += loss.item()
        rmse, mbe = evaluate(model, dl_va, dev)
        print(f"epoca {ep+1:3d}/{args.epochs} loss={tot/len(dl_tr):.4f} "
              f"val RMSE={rmse:.2f}C MBE={mbe:+.2f}C")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": best}, ck_last)
        if rmse < best:
            best = rmse
            torch.save(model.state_dict(), ck_best)

    model.load_state_dict(torch.load(ck_best, map_location=dev))
    rmse, mbe = evaluate(model, dl_te, dev)
    print(f"\n== TEST (hold-out): RMSE={rmse:.2f} C, MBE={mbe:+.2f} C ==")
    print(f"(referencia del paper: RMSE 1.92 C, MBE 0.01 C con ~20 anos de data)")


if __name__ == "__main__":
    main()
