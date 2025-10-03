import argparse, json, os, random, glob, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)

def find_images(root):
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, "**", e), recursive=True)
    return sorted(files)

def detect_mask_format(img_path, masks_root):
    """
    Return a normalized mask_info dict:
    """
    stem = Path(img_path).stem

    # Case A: composite single file
    cand_files = []
    for e in ("png","tif","tiff"):
        cand = os.path.abspath(os.path.join(masks_root, f"{stem}.{e}"))
        if os.path.exists(cand):
            cand_files.append(cand)
    if cand_files:
        return {"type": "composite", "paths": [cand_files[0]]}

    # Case B: multiple binary masks in a subfolder named <stem>
    multi = sorted(glob.glob(os.path.join(masks_root, stem, "*.*")))
    multi = [os.path.abspath(p) for p in multi]
    if multi:
        return {"type": "multiple", "paths": multi}

    # Case C: multiple masks matching <stem>_* in masks_root
    multi2 = sorted(glob.glob(os.path.join(masks_root, f"{stem}_*.*")))
    multi2 = [os.path.abspath(p) for p in multi2]
    if multi2:
        return {"type": "multiple", "paths": multi2}

    return {"type": "none", "paths": []}

def compute_ref_tile(image_paths, k=50, tile_size=256):
    # Simple average-color tile in LAB as a reference for Reinhard
    choose = image_paths if len(image_paths) <= k else random.sample(image_paths, k)
    lab_list = []
    for p in choose:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = tile_size / max(h, w)
        img = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))
        pad = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
        pad[:img.shape[0], :img.shape[1]] = img
        lab = cv2.cvtColor(pad, cv2.COLOR_RGB2LAB)
        lab_list.append(lab)
    if not lab_list:
        return None
    mean_lab = np.mean(np.stack(lab_list, 0), axis=0).astype(np.uint8)
    return mean_lab

def index_split(img_dir, mask_dir, desc="Indexing"):
    """Index image mask pairs under the given dirs."""
    img_paths = find_images(img_dir)
    records = []
    skipped = 0
    for p in tqdm(img_paths, desc=desc):
        det = detect_mask_format(p, mask_dir)
        if det["type"] == "none" or len(det["paths"]) == 0:
            skipped += 1
            continue
        records.append({"image": os.path.abspath(p), "mask_info": det})
    return records, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Raw Kaggle root containing train/ and test/ with original/ & mask/")
    ap.add_argument("--out_root", required=True, help="Output processed root")
    ap.add_argument("--val_frac", type=float, default=0.15, help="fraction of TRAIN to use for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # Expected layout:
    # <data_root>/train/original, <data_root>/train/mask
    # <data_root>/test/original,  <data_root>/test/mask
    train_img_dir  = os.path.join(args.data_root, "train", "original")
    train_mask_dir = os.path.join(args.data_root, "train", "mask")
    test_img_dir   = os.path.join(args.data_root, "test",  "original")
    test_mask_dir  = os.path.join(args.data_root, "test",  "mask")

    for d in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        if not os.path.isdir(d):
            raise SystemExit(f"Expected directory not found: {d}")

    # Index train (for train/val) and test separately
    train_records, train_skipped = index_split(train_img_dir, train_mask_dir, desc="Indexing (train)")
    test_records,  test_skipped  = index_split(test_img_dir,  test_mask_dir,  desc="Indexing (test)")

    if not train_records:
        raise SystemExit("No train image/mask pairs found.")
    if not test_records:
        print("Warning: no test image/mask pairs found.")

    # Train/Val split from TRAIN ONLY
    random.shuffle(train_records)
    n = len(train_records)
    n_val = int(round(n * args.val_frac))
    val_records   = train_records[:n_val]
    train_records = train_records[n_val:]

    # Write splits
    os.makedirs(os.path.join(args.out_root, "splits"), exist_ok=True)
    with open(os.path.join(args.out_root, "splits", "train.json"), "w") as f:
        json.dump(train_records, f, indent=2)
    with open(os.path.join(args.out_root, "splits", "val.json"), "w") as f:
        json.dump(val_records, f, indent=2)
    with open(os.path.join(args.out_root, "splits", "test.json"), "w") as f:
        json.dump(test_records, f, indent=2)

    # Reference tile from TRAIN images only
    ref_tile = compute_ref_tile([r["image"] for r in train_records], k=50)
    if ref_tile is not None:
        ref_path = os.path.join(args.out_root, "reference_lab.png")
        cv2.imwrite(ref_path, cv2.cvtColor(ref_tile, cv2.COLOR_LAB2BGR))
        print(f"Saved reference tile to {ref_path}")

    print(f"\nWrote splits to {os.path.join(args.out_root, 'splits')}")
    print(f"  train: {len(train_records)}")
    print(f"  val  : {len(val_records)} (from TRAIN)")
    print(f"  test : {len(test_records)} (from TEST)")
    if train_skipped:
        print(f"  skipped (train, no mask found): {train_skipped}")
    if test_skipped:
        print(f"  skipped (test,  no mask found): {test_skipped}")

if __name__ == "__main__":
    main()
