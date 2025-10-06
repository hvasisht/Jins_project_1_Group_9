import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- point this to the metrics you want to plot ---
metrics_csv = Path("runs/attr_resnet50/metrics.csv")   # change to resnet18 or any run

# infer output folder from CSV location
run_dir = metrics_csv.parent
df = pd.read_csv(metrics_csv)

train_df = df[df["split"] == "train"]
valid_df = df[df["split"] == "valid"]

# Loss
plt.figure()
plt.plot(train_df["epoch"], train_df["loss"], label="Train Loss")
plt.plot(valid_df["epoch"], valid_df["loss"], label="Valid Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{run_dir.name} Loss Curve"); plt.legend()
plt.savefig(run_dir / "loss_curve.png"); plt.close()

# Accuracy
plt.figure()
plt.plot(train_df["epoch"], train_df["acc"], label="Train Accuracy")
plt.plot(valid_df["epoch"], valid_df["acc"], label="Valid Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{run_dir.name} Accuracy Curve"); plt.legend()
plt.savefig(run_dir / "accuracy_curve.png"); plt.close()

# F1
plt.figure()
plt.plot(train_df["epoch"], train_df["f1"], label="Train F1")
plt.plot(valid_df["epoch"], valid_df["f1"], label="Valid F1")
plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.title(f"{run_dir.name} F1 Curve"); plt.legend()
plt.savefig(run_dir / "f1_curve.png"); plt.close()

print(f"✅ Plots saved in {run_dir}")
