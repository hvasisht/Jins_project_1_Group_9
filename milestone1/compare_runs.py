from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CHANGE THESE if your folders are named differently ---
RUNS = {
    "ResNet18 (e10)": Path("runs/attr_resnet18_e10/metrics.csv"),
    "ResNet50 (e5)" : Path("runs/attr_resnet50_e5/metrics.csv"),
}

# --- Output dir: new, separate, timestamped folder ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path(f"runs/compare_resnets_{stamp}")
OUT.mkdir(parents=True, exist_ok=True)

def read_split(df, split):
    d = df[df["split"] == split].copy()
    if "epoch" in d.columns:
        d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
        d = d.dropna(subset=["epoch"]).sort_values("epoch")
    return d

# -------- Overlay: Validation Accuracy --------
plt.figure()
for label, p in RUNS.items():
    if not p.exists():
        print(f"⚠️ Missing: {p} — skipping {label}")
        continue
    df = pd.read_csv(p)
    v = read_split(df, "valid")
    plt.plot(v["epoch"], v["acc"], label=label)
plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy — ResNet18 vs ResNet50")
plt.legend()
plt.savefig(OUT / "val_accuracy_overlay.png"); plt.close()

# -------- Overlay: Validation F1 --------
plt.figure()
for label, p in RUNS.items():
    if not p.exists(): continue
    df = pd.read_csv(p)
    v = read_split(df, "valid")
    plt.plot(v["epoch"], v["f1"], label=label)
plt.xlabel("Epoch"); plt.ylabel("Validation F1")
plt.title("Validation F1 — ResNet18 vs ResNet50")
plt.legend()
plt.savefig(OUT / "val_f1_overlay.png"); plt.close()

# -------- Bar charts: Test metrics (last row) --------
names, test_accs, test_f1s, test_losses = [], [], [], []
for label, p in RUNS.items():
    if not p.exists(): continue
    df = pd.read_csv(p)
    test = df.tail(1).iloc[0]
    names.append(label)
    test_losses.append(float(test["loss"]))
    test_accs.append(float(test["acc"]))
    test_f1s.append(float(test["f1"]))

# Accuracy bar
plt.figure()
plt.bar(range(len(names)), test_accs)
plt.xticks(range(len(names)), names)
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy — Comparison")
plt.tight_layout()
plt.savefig(OUT / "test_accuracy_bar.png"); plt.close()

# F1 bar
plt.figure()
plt.bar(range(len(names)), test_f1s)
plt.xticks(range(len(names)), names)
plt.ylabel("Test F1")
plt.title("Test F1 — Comparison")
plt.tight_layout()
plt.savefig(OUT / "test_f1_bar.png"); plt.close()

# -------- Summary table + “winner” --------
summary = pd.DataFrame({
    "model": names,
    "test_loss": test_losses,
    "test_acc":  test_accs,
    "test_f1":   test_f1s,
}).sort_values("test_acc", ascending=False)

# Simple scoring rule (adjust weights if you like)
w_acc, w_f1 = 0.5, 0.5
summary["score"] = w_acc*summary["test_acc"] + w_f1*summary["test_f1"]

winner = summary.sort_values("score", ascending=False).iloc[0]["model"]

summary.to_csv(OUT / "comparison_summary.csv", index=False)

# Also write a tiny markdown report for the professor
with open(OUT / "README.md", "w") as f:
    f.write("# ResNet18 vs ResNet50 — Comparison\n\n")
    f.write("## Test Metrics\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")
    f.write(f"**Scoring:** score = {w_acc} × accuracy + {w_f1} × F1\n\n")
    f.write(f"**Winner:** **{winner}**\n\n")
    f.write("Artifacts:\n")
    f.write("- val_accuracy_overlay.png\n")
    f.write("- val_f1_overlay.png\n")
    f.write("- test_accuracy_bar.png\n")
    f.write("- test_f1_bar.png\n")
    f.write("- comparison_summary.csv\n")

print(f"✅ Comparison saved to: {OUT}")
