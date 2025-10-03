
import argparse, os, glob, cv2, numpy as np
from tqdm import tqdm

def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    # If 3-channel (paletted/colored), reduce to label index
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m

def components_in_label(mask, label, min_area=10):
    binm = (mask == label).astype(np.uint8)
    num, comp = cv2.connectedComponents(binm)
    # count non-trivial comps
    k = 0
    for i in range(1, num):
        if (comp == i).sum() >= min_area:
            k += 1
    return k

def analyze_mask(path, max_labels=2000, min_area=10):
    m = load_mask(path)
    vals = np.unique(m)
    vals = vals[vals > 0]  # drop background 0
    n_unique = int(len(vals))
    # quick path: few unique values -> likely semantic
    if n_unique <= 10:
        # but still compute components per class to be sure
        comps = [components_in_label(m, v, min_area=min_area) for v in vals]
        mean_cc = float(np.mean(comps)) if comps else 0.0
        return {
            "path": path, "n_unique": n_unique, "mean_components_per_label": mean_cc,
            "frac_labels_multi_component": float(np.mean([c>1 for c in comps])) if comps else 0.0,
            "heuristic": "semantic_likely"
        }

    # otherwise, sample up to max_labels for speed
    if n_unique > max_labels:
        vals = np.random.RandomState(0).choice(vals, size=max_labels, replace=False)
    comps = [components_in_label(m, v, min_area=min_area) for v in vals]
    mean_cc = float(np.mean(comps)) if comps else 0.0
    frac_multi = float(np.mean([c>1 for c in comps])) if comps else 0.0

    # Decision rule:
    # - instance-labeled should have ~1 component per unique value and near-zero multi-component labels
    # - semantic-labeled will have few unique labels (handled above) OR many labels but many of them with multiple components (rare, but possible if mask uses arbitrary IDs for classes)
    # thresholds are conservative
    if (mean_cc <= 1.2) and (frac_multi <= 0.1):
        verdict = "instance_likely"
    else:
        verdict = "semantic_likely"

    return {
        "path": path,
        "n_unique": int(n_unique),
        "mean_components_per_label": mean_cc,
        "frac_labels_multi_component": frac_multi,
        "heuristic": verdict
    }

def main():
    ap = argparse.ArgumentParser(description="Audit composite masks to decide if they are instance-labeled or semantic-labeled.")
    ap.add_argument("--masks_dir", required=True, help="Directory containing composite mask PNGs")
    ap.add_argument("--glob", default="*.png", help="Glob for mask files (default: *.png)")
    ap.add_argument("--max_files", type=int, default=50, help="Sample up to N masks")
    ap.add_argument("--max_labels", type=int, default=2000, help="Max labels to sample per mask for speed")
    ap.add_argument("--min_area", type=int, default=10, help="Ignore connected components smaller than this area")
    ap.add_argument("--out_json", default="audit_masks_report.json", help="Output JSON path")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.masks_dir, args.glob)))
    if not files:
        raise SystemExit(f"No masks found in {args.masks_dir} with pattern {args.glob}")
    if len(files) > args.max_files:
        import random; random.seed(0)
        files = random.sample(files, args.max_files)

    rows = []
    for p in tqdm(files, desc="Auditing"):
        try:
            row = analyze_mask(p, max_labels=args.max_labels, min_area=args.min_area)
            rows.append(row)
        except Exception as e:
            rows.append({"path": p, "error": str(e)})

    # Aggregate verdict
    inst = sum(1 for r in rows if r.get("heuristic") == "instance_likely")
    sema = sum(1 for r in rows if r.get("heuristic") == "semantic_likely")
    verdict = "instance_likely" if inst >= sema else "semantic_likely"

    report = {
        "summary": {
            "files_analyzed": len(rows),
            "instance_votes": inst,
            "semantic_votes": sema,
            "dataset_verdict": verdict
        },
        "details": rows
    }

    with open(args.out_json, "w") as f:
        import json; json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}")
    print("SUMMARY:", report["summary"])

if __name__ == "__main__":
    main()
