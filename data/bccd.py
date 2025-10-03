import os, json, cv2, numpy as np, random
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

from .transforms import ReinhardNormalizer, macenko_normalize, to_uint8

def load_reference_lab(path):
    if path is None or (isinstance(path, str) and not os.path.isfile(path)):
        return None
    ref_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    return ref_rgb

def build_augs(img_size=512, aug_level="strong"):
    augs = []
    if aug_level in ["light", "strong"]:
        augs += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        augs += [A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5)]
        augs += [A.GaussNoise(p=0.3), A.MotionBlur(p=0.2)]
    if aug_level == "strong":
        augs += [A.ElasticTransform(p=0.2, alpha=1, sigma=50)]
        augs += [A.GridDistortion(p=0.2)]
    augs += [A.LongestMaxSize(max_size=img_size),
             A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101)]
    return A.Compose(augs)

def decode_masks(mask_info, segmentation='semantic'):
    """
    Returns list of instance binary masks (H, W) uint8 in {0,1}.
    Handles:
      - Composite instance masks (many unique labels)
      - Composite semantic masks (few labels): split by connected components per label.
    """
    import cv2, numpy as np
    if mask_info["type"] == "composite":
        p = mask_info["paths"][0]
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            return []
        if m.ndim == 3:
            # for paletted/colored PNGs: collapse to index
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

        labels = np.unique(m)
        labels = labels[labels > 0]  # drop background
        masks = []

        if segmentation=="semantic":
            for lab in labels:
                cls = (m == lab).astype(np.uint8)
                if cls.sum() == 0:
                    continue
                num, cc = cv2.connectedComponents(cls)
                for idx in range(1, num):
                    inst = (cc == idx).astype(np.uint8)
                    if inst.sum() > 0:
                        masks.append(inst)
        else:
            # INSTANCE-labeled composite (each value already an instance)
            for lab in labels:
                inst = (m == lab).astype(np.uint8)
                if inst.sum() > 0:
                    masks.append(inst)
        return masks

    elif mask_info["type"] == "multiple":
        out = []
        for p in mask_info["paths"]:
            mm = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if mm is None:
                continue
            mm = (mm > 0).astype(np.uint8)
            if mm.sum() > 0:
                out.append(mm)
        return out

    return []


def instances_to_targets(instances, erode_boundary=1, center_radius=3):
    if not instances:
        return None, None, None
    H, W = instances[0].shape
    inst_map = np.zeros((H,W), dtype=np.int32)
    for i, m in enumerate(instances, start=1):
        inst_map[m>0] = i
    # Foreground map
    cell = (inst_map>0).astype(np.uint8)
    # Boundary map
    kernel = np.ones((3,3), np.uint8)
    dil = cv2.dilate(inst_map.astype(np.uint8), kernel, iterations=1)
    ero = cv2.erode(inst_map.astype(np.uint8), kernel, iterations=1)
    boundary = ((dil!=ero) & (inst_map>0)).astype(np.uint8)
    if erode_boundary>0:
        boundary = cv2.dilate(boundary, kernel, iterations=erode_boundary)
    # Center: peaks of distance transform per instance
    center = np.zeros_like(cell)
    for lab in range(1, inst_map.max()+1):
        m = (inst_map==lab).astype(np.uint8)
        if m.sum()==0: continue
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        if dist.max() <= 0: continue
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        cv2.circle(center, (x,y), center_radius, 1, -1)
    return cell, boundary, center

class BCCDDataset(Dataset):
    def __init__(self, index_json, img_size=512, stain_norm="reinhard", reference_lab=None, augment="strong"):
        with open(index_json, "r") as f:
            self.records = json.load(f)
        self.img_size = img_size
        self.stain_norm = stain_norm
        self.reference_lab = reference_lab
        self.augs = build_augs(img_size, augment)

        self._reinhard = None
        if self.stain_norm == "reinhard" and self.reference_lab is not None:
            from .transforms import ReinhardNormalizer
            self._reinhard = ReinhardNormalizer.from_reference_lab(self.reference_lab)

    def __len__(self): return len(self.records)

    def _apply_stain_norm(self, img_rgb):
        if self.stain_norm == "none":
            return img_rgb
        if self.stain_norm == "reinhard" and self._reinhard is not None:
            return self._reinhard(img_rgb)
        if self.stain_norm == "macenko":
            from .transforms import macenko_normalize
            return macenko_normalize(img_rgb)
        return img_rgb

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_bgr = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        assert img_bgr is not None, f"Cannot read {rec['image']}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = self._apply_stain_norm(img_rgb)
    
        masks = decode_masks(rec["mask_info"])
        if len(masks) == 0:
            H, W = img_rgb.shape[:2]
            inst_map0 = np.zeros((H, W), np.int32)
            cell0 = np.zeros((H, W), np.uint8)
            boundary0 = np.zeros((H, W), np.uint8)
            center0 = np.zeros((H, W), np.uint8)
        else:
            # build an instance-labeled map FIRST (so we can transform it together)
            H, W = masks[0].shape
            inst_map0 = np.zeros((H, W), dtype=np.int32)
            for i, m in enumerate(masks, start=1):
                inst_map0[m > 0] = i
            # then cell/boundary/center
            cell0, boundary0, center0 = instances_to_targets(masks)
    
        # apply same albumentations to image + ALL masks, including inst_map
        aug = self.augs(image=img_rgb, masks=[cell0, boundary0, center0, inst_map0.astype(np.uint8)])
        img = aug["image"]
        cell, boundary, center, inst_map = aug["masks"]
        inst_map = inst_map.astype(np.int32)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC->CHW
        cell = cell.astype(np.float32)[None, ...]
        boundary = boundary.astype(np.float32)[None, ...]
        center = center.astype(np.float32)[None, ...]
        # return 'gt_inst' in network-input space
        return {
            "image": img,
            "cell": cell,
            "boundary": boundary,
            "center": center,
            "gt_inst": inst_map,  
            "meta": rec,
        }