import numpy as np

def compute_iou_dice(pred, gt):
    pred = (pred>0).astype(np.uint8)
    gt = (gt>0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum() + 1e-6
    iou = inter / union
    dice = 2*inter / (pred.sum() + gt.sum() + 1e-6)
    return float(iou), float(dice)

def match_instances(pred_inst, gt_inst, iou_thr=0.5):
    pred_ids = [i for i in range(1, pred_inst.max()+1)]
    gt_ids = [i for i in range(1, gt_inst.max()+1)]
    matches = []
    used_gt = set()
    for pid in pred_ids:
        p = (pred_inst==pid)
        best_iou, best_gid = 0.0, None
        for gid in gt_ids:
            if gid in used_gt: continue
            g = (gt_inst==gid)
            inter = np.logical_and(p,g).sum()
            union = np.logical_or(p,g).sum() + 1e-6
            iou = inter/union
            if iou > best_iou:
                best_iou, best_gid = iou, gid
        if best_iou >= iou_thr:
            matches.append((pid, best_gid, best_iou))
            used_gt.add(best_gid)
    tp = len(matches)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - len(used_gt)
    return matches, tp, fp, fn

def average_precision(recalls, precisions):
    # standard method: integrate precision envelope over recall
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return ap

def compute_map(pred_insts, pred_scores, gt_inst, iou_thresholds=None):
    """
    pred_insts: inst_map (HxW int) for predictions
    pred_scores: list of scores (float) for each instance (same order as 1..K)
    gt_inst: HxW int32 ground truth instance map
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)

    # Convert inst_map to masks
    masks = [(pred_insts==i).astype(np.uint8) for i in range(1, pred_insts.max()+1)]
    scores = pred_scores

    # Sort by score desc
    if len(scores) != len(masks):
        # Fallback: uniform scores
        scores = [1.0 for _ in masks]
    order = np.argsort(-np.array(scores))
    masks = [masks[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    aps = []
    for thr in iou_thresholds:
        g_used = set()
        tp = np.zeros(len(masks))
        fp = np.zeros(len(masks))
        for i, pm in enumerate(masks):
            best_iou = 0.0; best_gid = None
            for gid in range(1, gt_inst.max()+1):
                if gid in g_used: continue
                gm = (gt_inst==gid).astype(np.uint8)
                inter = np.logical_and(pm, gm).sum()
                union = np.logical_or(pm, gm).sum() + 1e-6
                iou = inter/union
                if iou > best_iou:
                    best_iou, best_gid = iou, gid
            if best_iou >= thr:
                tp[i] = 1; g_used.add(best_gid)
            else:
                fp[i] = 1
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        recalls = cum_tp / (gt_inst.max() + 1e-6)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
        ap = average_precision(recalls, precisions)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0

def _instances_from_map(inst_map):
    ms = []
    for lab in range(1, int(inst_map.max()) + 1):
        m = (inst_map == lab)
        if m.any():
            ms.append(m)
    return ms

def _iou_matrix(pred_masks, gt_masks):
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=float)
    P, G = len(pred_masks), len(gt_masks)
    ious = np.zeros((P, G), dtype=float)
    for i, pm in enumerate(pred_masks):
        p_area = pm.sum()
        if p_area == 0:
            continue
        for j, gm in enumerate(gt_masks):
            inter = np.logical_and(pm, gm).sum()
            if inter == 0:
                ious[i, j] = 0.0
            else:
                union = p_area + gm.sum() - inter
                ious[i, j] = inter / union if union > 0 else 0.0
    return ious

def _ap_at_thresh(inst_map, scores, gt_inst, thr):
    pred_masks = _instances_from_map(inst_map)
    gt_masks   = _instances_from_map(gt_inst)
    P, G = len(pred_masks), len(gt_masks)

    if P == 0 and G == 0:
        return 1.0
    if P == 0:
        return 0.0

    # normalize scores to the right length
    if scores is None or len(scores) != P:
        scores = [1.0] * P
    order = np.argsort(np.asarray(scores))[::-1]  # numpy array ok
    pred_masks = [pred_masks[i] for i in order]   # reorder by score

    ious = _iou_matrix(pred_masks, gt_masks)
    gt_taken = np.zeros(G, dtype=bool)

    tp, fp = np.zeros(P), np.zeros(P)
    for i in range(P):
        j_best = -1; best = 0.0
        for j in range(G):
            if not gt_taken[j] and ious[i, j] >= best:
                best = ious[i, j]; j_best = j
        if j_best >= 0 and best >= thr:
            tp[i] = 1.0; gt_taken[j_best] = True
        else:
            fp[i] = 1.0

    # precision–recall 
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (G if G > 0 else 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for k in range(mpre.size - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap

def compute_map_detail(inst_map, scores, gt_inst, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)
    aps = [_ap_at_thresh(inst_map, scores, gt_inst, float(t)) for t in thresholds]
    return {
        "AP_mean": float(np.mean(aps)) if aps else 0.0,
        "AP50": float(aps[0]) if aps else 0.0,
        "AP75": float(aps[5]) if len(aps) > 5 else 0.0,
        "AP_by_thresh": {f"{t:.2f}": float(a) for t, a in zip(thresholds, aps)},
    }
