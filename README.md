
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

# Milestone 2 – Celebrity Detection with YOLOv8

**Summary**
- Faces organized as `celeba_images/images_<IDENTITY>/*.jpg`
- One global identity→class map ensures the **same ID** across specs/blur/beard/bald
- 2×2 grid collages + YOLO labels (90/10 split), trained YOLOv8(s) @1024px
- Prediction script draws IDs **inside** boxes; val overlays show **GT (blue)** vs **Pred (green)**
- Final weights kept in repo
Step 2: YOLOv8 Training

Custom-trained YOLOv8 to detect multiple celebrities in one image.

Model outputs bounding boxes + celebrity ID labels.

Step 3: Post-Processing

Used clean_grid_preds.py to remove overlaps and fix bounding box duplicates.

Final predictions show clear grids with non-overlapping boxes and labels.

**How to run**

1) **Generate collages**
python milestone2/scripts/make_grids.py --n 1200 --rows 2 --cols 2 --w 1024 --h 1024 \
  --img_root celeba_images --id_map milestone2/data/id_to_class.json --out milestone2/outputs

# Step 2: Train YOLOv8
yolo detect train model=yolov8s.pt data=milestone2/data/celeb_id.yaml imgsz=1024 epochs=60 \
  project=milestone2/runs name=yolov8s_m2_clean

# Step 3: Predict on images (boxes + IDs)
python milestone2/scripts/predict_ids.py /path/to/image_or_folder   # saves to milestone2/preds/


# Step 4: Review GT vs Pred (validation set)
python milestone2/scripts/audit_val.py   # overlays in milestone2/val_review/

-------

Results

Milestone 1: ResNet50 achieved higher accuracy but ResNet18 was more lightweight.

Milestone 2: YOLOv8 successfully detected multiple celebrities per image with bounding boxes and IDs.

