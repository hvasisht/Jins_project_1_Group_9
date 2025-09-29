
# Jins Project 1 – Discriminative Deep Learning on CelebA (Milestone 1)

This repository contains the implementation and results for **Milestone 1** of the Discriminative Deep Learning project.  
The goal is to train and evaluate **ResNet18** and **ResNet50** models on the CelebA dataset for attribute classification.

---

## 📂 Repository Structure
- `train_celeba.py` → Training script for ResNet models  
- `plot_metrics.py` → Generate training/validation plots  
- `compare_runs.py` → Compare ResNet18 vs ResNet50 results  
- `plots_run.py` → Helper script for plotting multiple runs  
- `runs/` → Contains metrics, checkpoints, and plots  
  - `attr_resnet18_e10*` → ResNet18 logs, metrics, and curves  
  - `attr_resnet50_e5*` → ResNet50 logs, metrics, and curves  
  - `compare_resnets_*` → Comparison plots + summary CSV  

---

## ⚙️ How to Run

### Train Models
```bash
# ResNet18 (10 epochs)
python train_celeba.py --root ~/data/celeba --model resnet18 --epochs 10 --batch_size 64 --tag e10

# ResNet50 (5 epochs)
python train_celeba.py --root ~/data/celeba --model resnet50 --epochs 5 --batch_size 48 --tag e5

#Generate plots
python plot_metrics.py

#Compare models
python compare_runs.py
