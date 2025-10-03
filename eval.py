# eval.py 
import argparse, os, json
import numpy as np
import torch, cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from inspect import signature

from data.bccd import BCCDDataset, load_reference_lab
from models.uni_backbone import UNIBackbone
from models.segmentation_head import UNIInstSeg
from utils.postprocess import pred_to_instances
from utils.visualize import overlay_instances, draw_tp_fp_fn
from metrics import compute_iou_dice, compute_map, compute_map_detail


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--stain_norm", default="reinhard", choices=["none", "reinhard", "macenko"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=2)
    # Post-processing knobs (safe; passed only if supported by pred_to_instances)
    ap.add_argument("--pp_thr", type=float, default=0.5, help="foreground threshold")
    ap.add_argument("--pp_min_area", type=int, default=20, help="min instance area (pixels)")
    ap.add_argument("--pp_w_boundary", type=float, default=0.3, help="boundary weight in watershed")
    ap.add_argument("--pp_peak", type=int, default=None, help="peak footprint (if supported)")
    return ap.parse_args()


def make_loader(args):
    ref_lab = load_reference_lab(os.path.join(args.data_root, "reference_lab.png"))
    ds = BCCDDataset(
        os.path.join(args.data_root, "splits", f"{args.split}.json"),
        img_size=args.img_size,
        stain_norm=args.stain_norm,
        reference_lab=ref_lab,
        augment="none",   # no random augs at eval time, but resize/pad still applied
    )
    return ds, DataLoader(ds, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data
    ds, loader = make_loader(args)

    # Model
    backbone = UNIBackbone(device=device, use_uni=True)
    model = UNIInstSeg(backbone).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Prepare safe kwargs for pred_to_instances 
    pp_kwargs = {}
    sig = signature(pred_to_instances).parameters
    if "thr" in sig:           pp_kwargs["thr"] = args.pp_thr
    if "min_area" in sig:      pp_kwargs["min_area"] = args.pp_min_area
    if "w_boundary" in sig:    pp_kwargs["w_boundary"] = args.pp_w_boundary
    if "peak_footprint" in sig: pp_kwargs["peak_footprint"] = args.pp_peak

    ious, dices = [], []
    map_means, ap50s, ap75s = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        
        img = batch["image"][0].cpu().numpy().transpose(1, 2, 0)   # HWC float [0,1]
        img_rgb = (img * 255).clip(0, 255).astype(np.uint8)

        # GT instance map already transformed by the dataset 
        gt_inst = batch["gt_inst"].numpy()[0].astype(np.int32)

        # Forward
        with torch.no_grad():
            x = batch["image"].to(device, non_blocking=True)
            out = model(x)
            cell     = out["cell"][0, 0].detach().cpu().numpy()
            boundary = out["boundary"][0, 0].detach().cpu().numpy()
            center   = out["center"][0, 0].detach().cpu().numpy()

        # Instances 
        inst_map, scores = pred_to_instances(cell, boundary, center, **pp_kwargs)

        # Sanity: shapes must match
        H, W = gt_inst.shape
        assert inst_map.shape == (H, W), f"inst_map {inst_map.shape} vs gt_inst {gt_inst.shape}"
        assert img_rgb.shape[:2] == (H, W), f"img_rgb {img_rgb.shape[:2]} vs gt_inst {gt_inst.shape}"

        # Metrics
        pred_sem = (inst_map > 0).astype(np.uint8)
        gt_sem   = (gt_inst  > 0).astype(np.uint8)
        
        iou, dice = compute_iou_dice(pred_sem, gt_sem)
        map_detail = compute_map_detail(inst_map, scores, gt_inst)
        ious.append(iou)
        dices.append(dice)
        map_means.append(map_detail["AP_mean"])
        ap50s.append(map_detail["AP50"])
        ap75s.append(map_detail["AP75"])

        # Save qualitative overlays
       
        meta = batch["meta"][0] if isinstance(batch["meta"], (list, tuple)) else batch["meta"]
        img_path = meta.get("image")
        if isinstance(img_path, (list, tuple)): img_path = img_path[0]
        if isinstance(img_path, list):          img_path = img_path[0]
        base = os.path.splitext(os.path.basename(img_path if isinstance(img_path, str) else "sample"))[0]

        def contours_only(img_rgb, inst, color=(0,255,0), thickness=2):
            vis = img_rgb.copy()
            for lab in range(1, int(inst.max())+1):
                cnts,_ = cv2.findContours((inst==lab).astype(np.uint8),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, color, thickness)
            return vis
        gt_only   = contours_only(img_rgb, gt_inst, color=(255,0,0))   # red = GT
        pred_only = contours_only(img_rgb, inst_map, color=(0,255,0))  # green = Pred
        cv2.imwrite(os.path.join(vis_dir, f"{base}_gt_only.jpg"),   cv2.cvtColor(gt_only,  cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(vis_dir, f"{base}_pred_only.jpg"),  cv2.cvtColor(pred_only,  cv2.COLOR_RGB2BGR))
        
    # Aggregate & save metrics
    metrics = {
      "mean_IoU": float(np.mean(ious)) if ious else 0.0,
      "mean_Dice": float(np.mean(dices)) if dices else 0.0,
      "mAP_50_95": float(np.mean(map_means)) if map_means else 0.0,
      "AP50": float(np.mean(ap50s)) if ap50s else 0.0,
      "AP75": float(np.mean(ap75s)) if ap75s else 0.0,
      "num_images": int(len(ious)),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Markdown report
    report_path = os.path.join(args.out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# UNI-InstSeg on BCCD — Evaluation Report\n\n")
        f.write("## Quantitative Results\n\n")
        f.write("| Metric | Value |\n|---|---:|\n")
        f.write(f"| Mean IoU | {metrics['mean_IoU']:.4f} |\n")
        f.write(f"| Mean Dice | {metrics['mean_Dice']:.4f} |\n")
        f.write(f"| mAP@[0.50:0.95] | {metrics['mAP_50_95']:.4f} |\n")
        f.write(f"| AP@0.50 | {metrics['AP50']:.4f} |\n")
        f.write(f"| AP@0.75 | {metrics['AP75']:.4f} |\n\n")
        f.write(f"Evaluated on `{args.split}` split with {metrics['num_images']} images.\n\n")
        f.write("## Qualitative Examples\n\n")

    print(f"Wrote metrics to {os.path.join(args.out_dir, 'metrics.json')} and report to {report_path}")


if __name__ == "__main__":
    main()
