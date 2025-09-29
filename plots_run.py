from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt

# find all resnet runs (tagged or not)
run_csvs = [
    *glob.glob("runs/attr_resnet18*/metrics.csv"),
    *glob.glob("runs/attr_resnet50*/metrics.csv"),
]
if not run_csvs:
    print("No metrics found under runs/attr_resnet18*/ or runs/attr_resnet50*/"); exit(0)

report_dir = Path("report_plots"); report_dir.mkdir(exist_ok=True)

for csv_path in run_csvs:
    p = Path(csv_path)
    run_dir = p.parent
    run_name = run_dir.name  # e.g., attr_resnet18_e10

    df = pd.read_csv(p)
    tr = df[df["split"]=="train"].copy()
    va = df[df["split"]=="valid"].copy()

    # ensure numeric epochs
    for d in (tr, va):
        if "epoch" in d.columns:
            d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
            d.dropna(subset=["epoch"], inplace=True)

    def save_plot(x, y1, y2, ylabel, fname_suffix, title):
        plt.figure()
        plt.plot(tr[x], tr[y1], label=f"Train {ylabel}")
        plt.plot(va[x], va[y2], label=f"Valid {ylabel}")
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(f"{run_name} — {title}")
        plt.legend()
        out1 = run_dir / f"{fname_suffix}.png"
        out2 = report_dir / f"{run_name}_{fname_suffix}.png"
        plt.savefig(out1); plt.savefig(out2); plt.close()
        print(f"Saved: {out1} and {out2}")

    save_plot("epoch","loss","loss","Loss","loss_curve","Loss Curve")
    save_plot("epoch","acc","acc","Accuracy","accuracy_curve","Accuracy Curve")
    save_plot("epoch","f1","f1","F1","f1_curve","F1 Curve")

print("✅ Per-model plots saved in each run folder and in report_plots/")
