import numpy as np, cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import h_maxima

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def pred_to_instances(cell_logits, boundary_logits=None, center_logits=None,
                      thr=0.6, min_area=40, use_center=True, w_boundary=1.0,
                      peak_footprint=7, h_rel=0.2):
    cell_prob = sigmoid(cell_logits)
    # thr = skimage.filters.threshold_otsu(cell_prob)
    mask = (cell_prob >= thr).astype(np.uint8)
    if mask.sum()==0:
        return np.zeros_like(mask, dtype=np.int32), []

    # Slight opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Distance map on mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    if center_logits is not None and use_center:
        centers = sigmoid(center_logits)
        dist = dist + centers * dist.max()
        # suppress shallow maxima; h relative to max
        h = h_rel * (dist.max() + 1e-6)
        # h-maxima on a float image via integer trick (optional)
        dist_uint = (dist / (dist.max()+1e-6) * 65535).astype(np.uint16)
        peaks = h_maxima(dist_uint, h= int(h*65535/(dist.max()+1e-6)))
        markers = cv2.connectedComponents(peaks.astype(np.uint8))[1]
    else:
        # peak_local_max with a larger footprint
        coords = peak_local_max(dist, footprint=np.ones((peak_footprint,peak_footprint)), labels=mask)
        markers = np.zeros_like(mask, dtype=np.int32)
        for i, (y,x) in enumerate(coords, start=1):
            markers[y,x] = i

    if markers.max()==0:
        num, cc = cv2.connectedComponents(mask)
        return cc, [float(cell_prob[cc==i].mean()) for i in range(1, num)]

    # Elevation 
    elev = -dist
    if boundary_logits is not None:
        elev = elev + w_boundary * sigmoid(boundary_logits)

    labels = watershed(elev, markers, mask=mask)

    # Filter small
    inst_map = np.zeros_like(labels, dtype=np.int32); k=0; scores=[]
    for lab in range(1, labels.max()+1):
        comp = (labels==lab)
        area = int(comp.sum())
        if area < min_area: continue
        k += 1
        inst_map[comp] = k
        scores.append(float(cell_prob[comp].mean()))
    return inst_map, scores
