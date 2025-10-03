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

def draw_tp_fp_fn(img_rgb, pred_inst, gt_inst, iou_thr=0.3):
    
    vis = img_rgb.copy()

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum() + 1e-6
        return inter/union

    gt_ids = list(range(1, gt_inst.max()+1))
    pred_ids = list(range(1, pred_inst.max()+1))
    matched_gt = set()
    matched_pred = set()

    # Count different types of matches for diagnostics
    excellent_matches = []  # IoU > 0.5
    partial_matches = []    # IoU 0.3-0.5
    no_matches = []         # IoU < 0.3

    for pid in pred_ids:
        p = (pred_inst==pid)
        best = 0; best_gid = None
        for gid in gt_ids:
            if gid in matched_gt: continue
            g = (gt_inst==gid)
            s = iou(p, g)
            if s>best:
                best = s; best_gid = gid
        
        if best >= 0.5:
            # Excellent match - GREEN
            matched_pred.add(pid); matched_gt.add(best_gid)
            excellent_matches.append(pid)
            cnts,_ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0,255,0), 3)  # Green
        elif best >= 0.3:
            # Partial match - YELLOW (probably correct but imperfect)
            matched_pred.add(pid); matched_gt.add(best_gid)
            partial_matches.append(best)  # Store IoU for analysis
            cnts,_ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0,255,255), 3)  # Yellow
        else:
            # No good match - RED (likely false positive)
            no_matches.append(best)  # Store IoU for analysis
            cnts,_ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255,0,0), 3)  # Red

    # BLUE for False Negatives (missed ground truth)
    missed_gt = 0
    for gid in gt_ids:
        if gid not in matched_gt:
            g = (gt_inst==gid).astype(np.uint8)
            if g.sum() > 0:  # Only if ground truth instance exists
                cnts,_ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0,0,255), 3)  # Blue
                missed_gt += 1

    # Print diagnostic information
    print(f"Detection Analysis:")
    print(f"  Excellent matches (Green): {len(excellent_matches)}")
    print(f"  Partial matches (Yellow): {len(partial_matches)}")
    print(f"  False positives (Red): {len(no_matches)}")
    print(f"  Missed cells (Blue): {missed_gt}")
    if partial_matches:
        print(f"  Average partial IoU: {np.mean(partial_matches):.3f}")
    if no_matches:
        print(f"  Average FP IoU: {np.mean(no_matches):.3f}")

    return vis

def draw_tp_fp_fn_centroid(img_rgb, pred_inst, gt_inst, distance_thr=20):
    """
    Alternative TP/FP/FN based on centroid distance rather than IoU.
    Use this for very dense cell scenarios where IoU matching fails.
    """
    vis = img_rgb.copy()

    def get_centroid(mask):
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def dist(a, b):
        if a is None or b is None: return float('inf')
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # Get centroids
    pred_centroids = {}
    for pid in range(1, pred_inst.max()+1):
        mask = (pred_inst==pid).astype(np.uint8)
        if mask.sum() > 0:
            pred_centroids[pid] = get_centroid(mask)

    gt_centroids = {}
    for gid in range(1, gt_inst.max()+1):
        mask = (gt_inst==gid).astype(np.uint8)
        if mask.sum() > 0:
            gt_centroids[gid] = get_centroid(mask)

    # Match by centroid distance
    matched_gt = set()
    matched_pred = set()

    for pid, pc in pred_centroids.items():
        best_dist = float('inf')
        best_gid = None
        for gid, gc in gt_centroids.items():
            if gid in matched_gt: continue
            d = dist(pc, gc)
            if d < best_dist:
                best_dist = d; best_gid = gid

        if best_dist <= distance_thr:
            matched_pred.add(pid); matched_gt.add(best_gid)
            # GREEN for True Positives
            mask = (pred_inst==pid).astype(np.uint8)
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0,255,0), 3)
        else:
            # RED for False Positives
            mask = (pred_inst==pid).astype(np.uint8)
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255,0,0), 3)

    # BLUE for False Negatives
    for gid in gt_centroids:
        if gid not in matched_gt:
            mask = (gt_inst==gid).astype(np.uint8)
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0,0,255), 3)

    print(f"Centroid-based analysis (dist_thr={distance_thr}):")
    print(f"  TP: {len(matched_pred)}, FP: {len(pred_centroids)-len(matched_pred)}, FN: {len(gt_centroids)-len(matched_gt)}")
    
    return vis
