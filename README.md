
# Jins Project 1 – Discriminative Deep Learning on CelebA

This repository contains the implementation and results for Milestone 1 and Milestone 2 of the Discriminative Deep Learning project.
The goal is to train deep learning models for face attribute classification and celebrity detection using the CelebA dataset.

---

train_celeba.py        → Training script for ResNet models (Milestone 1)
plot_metrics.py        → Generate training/validation plots
compare_runs.py        → Compare ResNet18 vs ResNet50 results
plots_run.py           → Helper script for plotting multiple runs

milestone2/scripts/    → Scripts for YOLOv8 celebrity detection
   ├── face_crop.py         → Crop faces from CelebA dataset
   ├── make_collages.py     → Concatenate images to form grids
   ├── make_grid_collages.py→ Generate labeled grid datasets
   ├── clean_grid_preds.py  → Post-process predictions to remove overlaps

runs/                  → Logs, metrics, checkpoints (ignored in repo)
outputs/               → Model predictions, sample collages (ignored in repo)


---

Milestone 1 – Attribute Classification with ResNet

Trained ResNet18 (10 epochs) and ResNet50 (5 epochs) on CelebA attributes.

Compared training/validation accuracy and loss.

Plotted performance curves and compared model results.

How to run: 

# Train ResNet18 (10 epochs)
python train_celeba.py --root ~/data/celeba --model resnet18 --epochs 10 --batch_size 64 --tag e10

# Train ResNet50 (5 epochs)
python train_celeba.py --root ~/data/celeba --model resnet50 --epochs 5 --batch_size 48 --tag e5

# Generate plots
python plot_metrics.py

# Compare runs
python compare_runs.py

-----

Milestone 2 – Celebrity Detection with YOLOv8

Step 1: Dataset Preparation

Cropped celebrity faces using CelebA.

Concatenated images into grid collages (train/test).

Augmented data with random transformations to increase diversity.

Step 2: YOLOv8 Training

Custom-trained YOLOv8 to detect multiple celebrities in one image.

Model outputs bounding boxes + celebrity ID labels.

Step 3: Post-Processing

Used clean_grid_preds.py to remove overlaps and fix bounding box duplicates.

Final predictions show clear grids with non-overlapping boxes and labels.

How to run:

# Step 1: Generate collages
python milestone2/scripts/make_grid_collages.py

# Step 2: Train YOLOv8
yolo task=detect mode=train model=yolov8n.pt data=data/data.yaml epochs=10 imgsz=640

# Step 3: Predict on test set
yolo task=detect mode=predict model=milestone2/yolov8n_ft/weights/best.pt source=milestone2/data/yolo/images/test save=True

# Step 4: Clean predictions (remove overlaps)
python milestone2/scripts/clean_grid_preds.py

-------

Results

Milestone 1: ResNet50 achieved higher accuracy but ResNet18 was more lightweight.

Milestone 2: YOLOv8 successfully detected multiple celebrities per image with bounding boxes and IDs.

