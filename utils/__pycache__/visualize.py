import numpy as np, cv2, matplotlib.pyplot as plt

def overlay_instances(img_rgb, inst_map, alpha=0.35):
    img = img_rgb.copy()
    contours, _ = cv2.findContours((inst_map>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255,0,0), 1)
    # Random colors per instance
    out = img.copy()
    for lab in range(1, inst_map.max()+1):
        comp = (inst_map==lab).astype(np.uint8)
        color = np.random.RandomState(lab).randint(0,255, size=3).tolist()
        mask_col = np.zeros_like(img); mask_col[comp>0] = color
        out = cv2.addWeighted(out, 1.0, mask_col, alpha, 0)
    return out

def draw_tp_fp_fn(img_rgb, pred_inst, gt_inst, iou_thr=0.5):
    # Match predicted and GT by IoU and color code
    h, w = img_rgb.shape[:2]
    vis = img_rgb.copy()

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum() + 1e-6
        return inter/union

    gt_ids = list(range(1, gt_inst.max()+1))
    pred_ids = list(range(1, pred_inst.max()+1))
    matched_gt = set()
    matched_pred = set()

    for pid in pred_ids:
        p = (pred_inst==pid)
        best = 0; best_gid = None
        for gid in gt_ids:
            if gid in matched_gt: continue
            g = (gt_inst==gid)
            s = iou(p, g)
            if s>best:
                best = s; best_gid = gid
        if best >= iou_thr:
            matched_pred.add(pid); matched_gt.add(best_gid)
            # green contour for TP
            cnts,_ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0,255,0), 2)
        else:
            # orange for FP
            cnts,_ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0,165,255), 2)

    # red for FN
    for gid in gt_ids:
        if gid not in matched_gt:
            g = (gt_inst==gid)
            cnts,_ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255,0,0), 2)

    return vis
