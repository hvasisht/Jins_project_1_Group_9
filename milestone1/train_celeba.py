# CelebA (Kaggle CSV layout) trainer for: resnet18, resnet50, inception_v3, unet
import argparse, time, csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

# ---------- device ----------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def count_params(model): return sum(p.numel() for p in model.parameters())

def metrics_np(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    f1  = f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)
    return acc, f1

# ---------- dataset (Kaggle CSVs) ----------
class CelebAAttrCSV(Dataset):
    """Needs: list_attr_celeba.csv, list_eval_partition.csv, img_align_celeba/"""
    def __init__(self, root, split="train", transform=None, limit=None):
        self.root = Path(root)
        self.transform = transform
        attr = self.root / "list_attr_celeba.csv"
        part = self.root / "list_eval_partition.csv"
        imgs = self.root / "img_align_celeba"
        if not (attr.exists() and part.exists() and imgs.exists()):
            raise FileNotFoundError("Expected CSVs and img_align_celeba/ under --root")

        attr_df = pd.read_csv(attr)
        part_df = pd.read_csv(part)
        attr_df.columns = [c.strip() for c in attr_df.columns]
        part_df.columns = [c.strip() for c in part_df.columns]

        col_img_attr = "image_id" if "image_id" in attr_df.columns else attr_df.columns[0]
        col_img_part = "image_id" if "image_id" in part_df.columns else part_df.columns[0]
        col_part     = "partition" if "partition" in part_df.columns else part_df.columns[1]

        df = attr_df.merge(part_df[[col_img_part, col_part]],
                           left_on=col_img_attr, right_on=col_img_part, how="inner")
        df.rename(columns={col_img_attr:"image_id", col_part:"partition"}, inplace=True)

        split_map = {"train":0, "valid":1, "val":1, "test":2}
        df = df[df["partition"] == split_map[split]].reset_index(drop=True)

        if limit and limit > 0:
            df = df.iloc[:limit].copy()

        self.attr_cols = [c for c in df.columns if c not in ["image_id","partition", col_img_part]]
        self.df = df
        self.img_dir = imgs

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.img_dir / r["image_id"]).convert("RGB")
        if self.transform: img = self.transform(img)
        # {-1,1} -> {0,1}
        y = torch.tensor([(1 if r[c] == 1 else 0) for c in self.attr_cols], dtype=torch.float32)
        return img, y

# ---------- simple U-Net encoder classifier ----------
class UNetEncoderClassifier(nn.Module):
    def __init__(self, num_classes=40, in_ch=3, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, 1, 1), nn.ReLU(True),
                                  nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, 1, 1), nn.ReLU(True),
                                  nn.Conv2d(base*2, base*2, 3, 1, 1), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, 1, 1), nn.ReLU(True),
                                  nn.Conv2d(base*4, base*4, 3, 1, 1), nn.ReLU(True))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, 1, 1), nn.ReLU(True),
                                  nn.Conv2d(base*8, base*8, 3, 1, 1), nn.ReLU(True))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(base*8, num_classes)
    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.enc4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

# ---------- model factory ----------
def build_model(name, num_outputs=40):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_outputs)
        return m, 224, False
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_outputs)
        return m, 224, False
    if name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_outputs)
        m.fc = nn.Linear(m.fc.in_features, num_outputs)
        return m, 299, True
    if name == "unet":
        return UNetEncoderClassifier(num_classes=num_outputs), 224, False
    raise ValueError(f"Unknown model: {name}")

# ---------- train/eval ----------
def train_one_epoch(model, loader, loss_fn, opt, device, use_inception=False):
    model.train()
    losses, ys, ps = [], [], []
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        if use_inception:
            out, aux = model(x)
            loss = loss_fn(out, y) + 0.4*loss_fn(aux, y)
        else:
            out = model(x)
            loss = loss_fn(out, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            p = (torch.sigmoid(out) > 0.5).int().cpu().numpy()
            ys.append(y.int().cpu().numpy()); ps.append(p)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc, f1 = metrics_np(y_true, y_pred)
    return float(sum(losses)/len(losses)), acc, f1

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, ys, ps = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y); losses.append(loss.item())
        p = (torch.sigmoid(out) > 0.5).int().cpu().numpy()
        ys.append(y.int().cpu().numpy()); ps.append(p)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc, f1 = metrics_np(y_true, y_pred)
    return float(sum(losses)/len(losses)), acc, f1

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18","resnet50","inception_v3","unet"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_train", type=int, default=0)  # 0 = full
    args = ap.parse_args()

    device = get_device()
    print("Using device:", device)

    # model + image size
    model, img_size, use_inception = build_model(args.model, num_outputs=40)
    model.to(device)

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # data
    train_set = CelebAAttrCSV(args.root, "train", transform=train_tf,
                              limit=(args.limit_train if args.limit_train>0 else None))
    valid_set = CelebAAttrCSV(args.root, "valid", transform=test_tf)
    test_set  = CelebAAttrCSV(args.root, "test",  transform=test_tf)

    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dl_valid = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # loss/opt
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # logging
    out_dir = Path("runs") / f"attr_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","split","loss","acc","f1","time_sec"])

    # train
    best_f1, best_path = -1.0, out_dir / "best.pt"
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        t_ep = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, dl_train, loss_fn, opt, device, use_inception)
        va_loss, va_acc, va_f1 = evaluate(model, dl_valid, loss_fn, device)
        dt = time.time() - t_ep
        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep,"train",f"{tr_loss:.4f}",f"{tr_acc:.4f}",f"{tr_f1:.4f}",f"{dt:.2f}"])
            w.writerow([ep,"valid",f"{va_loss:.4f}",f"{va_acc:.4f}",f"{va_f1:.4f}",f"{dt:.2f}"])
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {ep:02d} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"Valid loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} | {dt:.1f}s")

    total_time = time.time() - t0

    # test best
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, te_f1 = evaluate(model, dl_test, loss_fn, device)
    with open(metrics_csv, "a", newline="") as f:
        csv.writer(f).writerow(["best","test",f"{te_loss:.4f}",f"{te_acc:.4f}",f"{te_f1:.4f}",f"{total_time:.2f}"])

    size_mb = Path(best_path).stat().st_size/(1024*1024)
    params_m = count_params(model)/1e6
    print(f"\n=== DONE {args.model} ===")
    print(f"Total train time: {total_time:.1f}s | Test acc: {te_acc:.4f} | Test f1: {te_f1:.4f}")
    print(f"Params: {params_m:.2f}M | best.pt size: {size_mb:.2f} MB")
    print(f"Metrics: {metrics_csv}")

if __name__ == "__main__":
    main()
