# ResNet18 vs ResNet50 — Comparison

## Test Metrics
| model          |   test_loss |   test_acc |   test_f1 |   score |
|:---------------|------------:|-----------:|----------:|--------:|
| ResNet50 (e5)  |      0.1938 |     0.9133 |    0.8022 | 0.85775 |
| ResNet18 (e10) |      0.1997 |     0.9118 |    0.8039 | 0.85785 |

**Scoring:** score = 0.5 × accuracy + 0.5 × F1

**Winner:** **ResNet18 (e10)**

Artifacts:
- val_accuracy_overlay.png
- val_f1_overlay.png
- test_accuracy_bar.png
- test_f1_bar.png
- comparison_summary.csv
