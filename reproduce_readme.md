# UNI on BCCD 

This repo demonstrates how to repurpose the **UNI** pathology foundation model as a backbone for **cell instance segmentation** on the BCCD dataset (blood cell images with instance masks).

> **Key idea:** Use the ViT-L/16 encoder from UNI (pretrained on >100M histopathology patches) as a frozen (then partially unfrozen) backbone and add a lightweight decoder + multi-head outputs (cell mask, boundary, and center/seedness). Instances are reconstructed with marker-controlled watershed.

---

## 1. Environment

```bash
conda create -n segtorch python=3.10 -y
conda activate segtorch
pip install -r requirements.txt
```

> **UNI weights access**: You must accept the model license and enable access on Hugging Face.
Download the model from Huggingface
https://huggingface.co/MahmoodLab/UNI

## 2. Dataset: BCCD with masks

Download the dataset from Kaggle:
- https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask


Prepare splits and index:
```bash
python scripts/prepare_bccd.py --data_root data/bccd/raw --out_root data/bccd --val_frac 0.15  --seed 42
```

This script tries to auto-detect mask format:
- **Composite mask**: a single PNG where each instance has a unique integer (1..K).
- **Multiple binary masks**: a folder of binary PNGs per instance.

It also computes a reference tile for stain normalization (if enabled), and writes JSON index files in `data/bccd/splits/{train,val,test}.json`.

---

## 3. Train

```bash
python train.py   --data_root data/bccd   --backbone uni   --img_size 512   --batch_size 8   --epochs 50   --lr_head 1e-4   --lr_backbone 1e-5   --freeze_backbone_epochs 5   --stain_norm reinhard   --augment strong   --out_dir runs/exp001
```

Notes:
- Mixed precision is on by default.
- `--backbone` can be `uni` (preferred) or `dino_vitl16` (fallback).
- `--stain_norm` can be `none|reinhard|macenko`.

---

## 4. Evaluate & Visualize

```bash
python eval.py   --data_root data/bccd   --checkpoint runs/exp001/checkpoints/best.pt   --split test   --out_dir runs/exp001/eval_test
```

This computes **IoU**, **Dice**, and **mAP** (COCO-like over IoU=0.50:0.95) and writes a markdown report with qualitative overlays to:
```
runs/exp001/eval_test/report.md
```

---

## 5. Model details

- **Backbone**: UNI ViT-L/16 with `dynamic_img_size=True`, returning token maps → reshaped to a 2D feature map.
- **Decoder**: Lightweight upsampling path (Conv + GN + GELU + bilinear) repeated to full resolution.
- **Heads**: 
  - `cell`: foreground probability
  - `boundary`: thin boundary probability
  - `center`: seedness probability to seed watershed
- **Loss**: Dice+BCE for `cell`, BCE for `boundary` and `center`, with class-balanced weights.
- **Post-processing**: threshold → distance transform + `center` peaks → marker-controlled watershed (optionally use boundary map as barrier).


## 6. Citations

- UNI (Nature Medicine 2024): Chen *et al.*, *Towards a general-purpose foundation model for computational pathology*.  
- UNI repo loading via `timm` & HF hub (KatherLab/uni fork).  
- Stain normalization (Macenko / Reinhard).  
- Instance splitting via watershed; inspiration from HoVer-Net boundary-aware design (without hover vectors).

**See `report.md` produced after evaluation for results and qualitative examples.**
