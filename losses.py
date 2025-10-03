# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        # logits: (B,1,H,W), targets: (B,1,H,W) in {0,1}
        p = torch.sigmoid(logits)
        num = 2.0 * (p * targets).sum(dim=(1,2,3)) + self.eps
        den = (p**2 + targets**2).sum(dim=(1,2,3)) + self.eps
        return 1.0 - (num / den).mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.eps=eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p*targets).sum(dim=(1,2,3))
        fp = (p*(1-targets)).sum(dim=(1,2,3))
        fn = ((1-p)*targets).sum(dim=(1,2,3))
        tversky = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps)
        return 1.0 - tversky.mean()

def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0, weight=None, eps=1e-8):
    # weight is a per-pixel weight map (broadcastable)
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob*targets + (1-prob)*(1-targets)
    loss = (alpha * (1 - p_t).clamp(min=eps).pow(gamma) * ce)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

def bce_with_logits_weighted(logits, targets, weight=None, pos_weight=None):
    # 'weight' is per-pixel; 'pos_weight' is global (tensor scalar)
    return F.binary_cross_entropy_with_logits(
        logits, targets, weight=weight, pos_weight=pos_weight, reduction="mean"
    )

def make_pixel_weights(boundary, center, lam_b=1.5, lam_c=2.0):
    """
    boundary, center: (B,1,H,W) binary targets
    returns per-pixel weights w = 1 + lam_b*boundary + lam_c*center
    """
    return 1.0 + lam_b*boundary + lam_c*center

# multi-head loss
class MultiHeadSegLoss(nn.Module):
    """
    loss = λ_cell*(Dice + BCE(weight_map)) +
           λ_boundary*(0.5*BCE(pos_weight=2) + 0.5*Dice or Tversky) +
           λ_center*(Focal-BCE or BCE(pos_weight=5))

    Options:
      - use_tversky_for_boundary: use Tversky instead of Dice on boundary
      - dynamic_weights: Kendall & Gal uncertainty weighting across heads
    """
    def __init__(self,
                 w_cell=1.0, w_boundary=1.0, w_center=1.0,
                 lam_b_for_cell=1.5, lam_c_for_cell=2.0,
                 boundary_pos_weight=2.0, center_pos_weight=5.0,
                 use_focal_center=True, focal_alpha=0.25, focal_gamma=2.0,
                 use_tversky_for_boundary=False, tversky_alpha=0.3, tversky_beta=0.7,
                 dynamic_weights=False):
        super().__init__()
        self.w_cell = w_cell
        self.w_boundary = w_boundary
        self.w_center = w_center

        self.lam_b_for_cell = lam_b_for_cell
        self.lam_c_for_cell = lam_c_for_cell

        self.boundary_pos_weight = torch.tensor(boundary_pos_weight)
        self.center_pos_weight = torch.tensor(center_pos_weight)

        self.use_focal_center = use_focal_center
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.boundary_dice = SoftDiceLoss()
        self.boundary_tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.cell_dice = SoftDiceLoss()

        self.use_tversky_for_boundary = use_tversky_for_boundary

        self.dynamic_weights = dynamic_weights
        if dynamic_weights:
            # Kendall & Gal uncertainty weighting
            self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, out, targets):
        """
        out: dict with logits
        targets: dict with binary targets in {0,1}: same keys
        """
        device = out["cell"].device
        # move pos_weight tensors properly
        pw_b = self.boundary_pos_weight.to(device)
        pw_c = self.center_pos_weight.to(device)

        # pixel weight map for cell (emphasize separation pixels)
        w_map = make_pixel_weights(targets["boundary"], targets["center"],
                                   self.lam_b_for_cell, self.lam_c_for_cell)

        # cell: Dice + BCE(weight map)
        loss_cell = self.cell_dice(out["cell"], targets["cell"]) + \
                    bce_with_logits_weighted(out["cell"], targets["cell"], weight=w_map)

        # boundary: BCE(pos_weight) + Dice/Tversky
        loss_boundary = 0.5 * bce_with_logits_weighted(out["boundary"], targets["boundary"],
                                                       pos_weight=pw_b)
        if self.use_tversky_for_boundary:
            loss_boundary = loss_boundary + 0.5 * self.boundary_tversky(out["boundary"], targets["boundary"])
        else:
            loss_boundary = loss_boundary + 0.5 * self.boundary_dice(out["boundary"], targets["boundary"])

        # center: Focal-BCE or BCE(pos_weight)
        if self.use_focal_center:
            loss_center = focal_bce_with_logits(out["center"], targets["center"],
                                                alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss_center = bce_with_logits_weighted(out["center"], targets["center"],
                                                   pos_weight=pw_c)

        losses = torch.stack([loss_cell, loss_boundary, loss_center])

        if not self.dynamic_weights:
            total = self.w_cell*loss_cell + self.w_boundary*loss_boundary + self.w_center*loss_center
            return total, {
                "loss_total": total.item(),
                "loss_cell": loss_cell.item(),
                "loss_boundary": loss_boundary.item(),
                "loss_center": loss_center.item()
            }
        else:
            # Kendall & Gal: sum exp(-s_i)*L_i + s_i, s_i = log_var_i
            inv_vars = torch.exp(-self.log_vars)
            total = (inv_vars * losses).sum() + self.log_vars.sum()
            return total, {
                "loss_total": total.item(),
                "loss_cell": loss_cell.item(),
                "loss_boundary": loss_boundary.item(),
                "loss_center": loss_center.item(),
                "log_vars": self.log_vars.detach().cpu().tolist()
            }
