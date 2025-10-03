import argparse, os, json, random, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.bccd import BCCDDataset, load_reference_lab
from models.uni_backbone import UNIBackbone
from models.segmentation_head import UNIInstSeg
from losses import MultiHeadSegLoss

def set_seed(seed=17):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="processed root with splits/*.json and reference_lab.png")
    ap.add_argument("--backbone", default="uni", choices=["uni"])
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=5)
    ap.add_argument("--stain_norm", default="reinhard", choices=["none","reinhard","macenko"])
    ap.add_argument("--augment", default="strong", choices=["none","light","strong"])
    ap.add_argument("--out_dir", default="runs/exp001")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad_clip", type=float, default=0.0)
    return ap.parse_args()

def make_loader(split_json, args, training=True):
    ref_lab = load_reference_lab(os.path.join(args.data_root, "reference_lab.png"))
    ds = BCCDDataset(
        split_json,
        img_size=args.img_size,
        stain_norm=args.stain_norm,
        reference_lab=ref_lab,
        augment=(args.augment if training else "none"),
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size if training else 1,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=training,
    )

def build_optimizer_param_groups(model, lr_head, lr_backbone):
    """Return param groups: heads/decoder at lr_head; unfrozen backbone at lr_backbone."""
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone.encoder.blocks"):
            backbone_params.append(p)   # only newly unfrozen blocks will have requires_grad=True
        else:
            head_params.append(p)
    return [
        {"params": head_params, "lr": lr_head},
        {"params": backbone_params, "lr": lr_backbone},
    ]

def main():
    args = parse_args()
    set_seed(17)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Model ---
    backbone = UNIBackbone(device=device, use_uni=(args.backbone == "uni"))
    model = UNIInstSeg(backbone).to(device)

    # Freeze backbone initially
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Optimizer (heads only to start)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_head, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Loss function
    crit = MultiHeadSegLoss(
        w_cell=1.0, w_boundary=1.0, w_center=1.0,
        lam_b_for_cell=1.5, lam_c_for_cell=2.0,
        boundary_pos_weight=2.0, center_pos_weight=5.0,
        use_focal_center=True, focal_alpha=0.25, focal_gamma=2.0,
        use_tversky_for_boundary=False,  # set True to try Focal-Tversky variant (see earlier message)
        dynamic_weights=True
    ).to(device)

    # Data
    tr_loader = make_loader(os.path.join(args.data_root, "splits", "train.json"), args, training=True)
    va_loader = make_loader(os.path.join(args.data_root, "splits", "val.json"), args, training=False)

    best_val = float("inf"); best_path = None

    for epoch in range(1, args.epochs + 1):
        # Unfreeze schedule
        if epoch == args.freeze_backbone_epochs + 1:
            # Unfreeze UNI blocks (last 6 for sharper boundaries)
            for n, p in model.backbone.encoder.blocks[-6:].named_parameters():
                p.requires_grad = True
            # Rebuild optimizer with param groups to honor lr_backbone
            param_groups = build_optimizer_param_groups(model, args.lr_head, args.lr_backbone)
            optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        
        model.train()
        tr_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            targets = {
                "cell": batch["cell"].to(device, non_blocking=True),
                "boundary": batch["boundary"].to(device, non_blocking=True),
                "center": batch["center"].to(device, non_blocking=True),
            }

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss, loss_dict = crit(outputs, targets)

            if args.grad_clip > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim); scaler.update()
            else:
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()

            tr_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lc=f"{loss_dict['loss_cell']:.3f}",
                             lb=f"{loss_dict['loss_boundary']:.3f}",
                             lz=f"{loss_dict['loss_center']:.3f}")

        tr_loss /= max(1, len(tr_loader))

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                targets = {
                    "cell": batch["cell"].to(device, non_blocking=True),
                    "boundary": batch["boundary"].to(device, non_blocking=True),
                    "center": batch["center"].to(device, non_blocking=True),
                }
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss, _ = crit(outputs, targets)
                val_loss += loss.item()
        val_loss /= max(1, len(va_loader))
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        # Checkpoints
        ckpt_dir = os.path.join(args.out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)

    print(f"Best val loss: {best_val:.4f} at {best_path}")

if __name__ == "__main__":
    main()
