from pathlib import Path
import pandas as pd

def read_test_row(metrics_path: Path):
    df = pd.read_csv(metrics_path)
    # last row is the test results (best model)
    test = df.tail(1).iloc[0]
    return {
        "test_loss": float(test["loss"]),
        "test_acc": float(test["acc"]),
        "test_f1": float(test["f1"]),
    }

rows = []
for model in ["resnet18", "resnet50"]:
    p = Path(f"runs/attr_{model}/metrics.csv")
    if p.exists():
        res = read_test_row(p)
        rows.append({"model": model, **res})
    else:
        rows.append({"model": model, "test_loss": None, "test_acc": None, "test_f1": None})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("runs/resnet_compare.csv", index=False)
print("\n✅ Comparison saved: runs/resnet_compare.csv")
