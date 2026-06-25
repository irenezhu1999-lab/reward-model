#!/usr/bin/env python3
"""
冷启动 Fluidity 奖励模型 — 纯 Python 线性回归，无需 sklearn。
输入：光流特征 + 缺陷 flag
输出：fluidity 分数 1-10

使用方法：
  python train_fluidity_model.py              # 训练 + LOO-CV 评估
  python train_fluidity_model.py --predict <mp4_path>  # 对新视频打分
"""
import json, re, sys
from pathlib import Path

ANNOTATIONS = Path("D:/AI Magic 效果/reward_model/annotations.jsonl")
MODEL_OUT = Path("D:/AI Magic 效果/reward_model/fluidity_model_weights.json")

FEATURE_NAMES = [
    "bias", "mean_flow", "peak_flow", "peak_ratio",
    "motion_frames", "uniform_velocity", "motion_stutter", "unnatural_trajectory",
    "motion_freeze", "joint_isolation",
]

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(ann: dict) -> list[float] | None:
    notes = ann.get("notes") or ""
    m = re.search(r"mean=([\d.]+) peak=([\d.]+) motion_frames=(\d+)", notes)
    if not m:
        return None
    mean_f = float(m.group(1))
    peak_f = float(m.group(2))
    motion_n = int(m.group(3))
    return [
        1.0,  # bias
        mean_f,
        peak_f,
        peak_f / (mean_f + 1e-6),
        float(motion_n),
        float(ann["defect_flags"].get("uniform_velocity", False)),
        float(ann["defect_flags"].get("motion_stutter", False)),
        float(ann["defect_flags"].get("unnatural_trajectory", False)),
        float(ann["defect_flags"].get("motion_freeze", False)),
        float(ann["defect_flags"].get("joint_isolation", False)),
    ]

def extract_features_from_flow(mags, defects: dict, fps: float = 24.0) -> list[float]:
    import numpy as np
    smooth = np.convolve(mags, np.ones(3)/3, mode="same")
    mean_mag = float(smooth.mean())
    peak_mag = float(smooth.max())
    peak_idx = int(np.argmax(smooth))
    mean_thresh = mean_mag * 1.5
    motion_frames = int(np.sum(smooth > mean_thresh))
    return [
        1.0,
        mean_mag,
        peak_mag,
        peak_mag / (mean_mag + 1e-6),
        float(motion_frames),
        float(defects.get("uniform_velocity", False)),
        float(defects.get("motion_stutter", False)),
        float(defects.get("unnatural_trajectory", False)),
        float(defects.get("motion_freeze", False)),
        float(defects.get("joint_isolation", False)),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Pure-python linear algebra
# ─────────────────────────────────────────────────────────────────────────────

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_mul(A, B):
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]

def mat_add_eye(M, lam=0.1):
    for i in range(len(M)):
        M[i][i] += lam
    return M

def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for row in range(col + 1, n):
            f = M[row][col] / M[col][col]
            for k in range(col, n + 1):
                M[row][k] -= f * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x

def fit(X, y, lam=0.1):
    Xt = transpose(X)
    XtX = mat_add_eye(mat_mul(Xt, X), lam)
    Xty = [dot(row, y) for row in Xt]
    return solve(XtX, Xty)

def predict(w, x):
    return max(1.0, min(10.0, dot(w, x)))

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(template_filter: str = None):
    with open(ANNOTATIONS, encoding="utf-8") as f:
        anns = [json.loads(l) for l in f if l.strip()]

    X, y, ids, templates = [], [], [], []
    for a in anns:
        if a["scores"]["fluidity"] is None:
            continue
        feats = extract_features(a)
        if feats is None:
            continue
        tmpl = "patronus" if "patronus" in a["video_id"] else "pet_greeting"
        if template_filter and tmpl != template_filter:
            continue
        X.append(feats)
        y.append(float(a["scores"]["fluidity"]))
        ids.append(a["video_id"])
        templates.append(tmpl)
    return X, y, ids, templates

# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate
# ─────────────────────────────────────────────────────────────────────────────

def _loo_cv(X, y, ids):
    n = len(X)
    preds = []
    for i in range(n):
        X_tr = [X[j] for j in range(n) if j != i]
        y_tr = [y[j] for j in range(n) if j != i]
        w = fit(X_tr, y_tr)
        preds.append(predict(w, X[i]))
    errors = [abs(p - t) for p, t in zip(preds, y)]
    mae = sum(errors) / n
    baseline_mae = sum(abs(yi - sum(y)/n) for yi in y) / n
    within_1 = sum(1 for e in errors if e <= 1.0)
    return preds, errors, mae, baseline_mae, within_1


def train_and_eval():
    X, y, ids, templates = load_dataset()
    n = len(X)
    print(f"Dataset: {n} samples, fluidity range {int(min(y))}-{int(max(y))}")

    # Overall LOO-CV
    preds, errors, mae, baseline_mae, within_1 = _loo_cv(X, y, ids)
    print(f"\nLOO-CV MAE:  {mae:.2f}  (baseline: {baseline_mae:.2f})")
    print(f"Within ±1:   {within_1}/{n} = {within_1/n*100:.0f}%")

    # Stratified by template
    print("\nStratified LOO-CV:")
    for tmpl in ("patronus", "pet_greeting"):
        idx = [i for i, t in enumerate(templates) if t == tmpl]
        if not idx:
            continue
        Xs = [X[i] for i in idx]
        ys = [y[i] for i in idx]
        is_ = [ids[i] for i in idx]
        _, errs, tmpl_mae, tmpl_base, tmpl_w1 = _loo_cv(Xs, ys, is_)
        print(f"  {tmpl:<14} N={len(idx):3d}  MAE={tmpl_mae:.2f}  "
              f"baseline={tmpl_base:.2f}  within±1={tmpl_w1/len(idx)*100:.0f}%")

    # Train final model per template + save
    all_weights = {}
    for tmpl in ("patronus", "pet_greeting"):
        idx = [i for i, t in enumerate(templates) if t == tmpl]
        if not idx:
            continue
        Xs = [X[i] for i in idx]
        ys = [y[i] for i in idx]
        _, errs, tmpl_mae, _, _ = _loo_cv(Xs, ys, [ids[i] for i in idx])
        w = fit(Xs, ys)
        all_weights[tmpl] = {"weights": w, "n_train": len(idx), "loo_cv_mae": round(tmpl_mae, 3)}

    print("\nModel weights (patronus):")
    for name, coef in zip(FEATURE_NAMES, all_weights["patronus"]["weights"]):
        bar = "+" * int(abs(coef) * 2) if coef > 0 else "-" * int(abs(coef) * 2)
        print(f"  {name:<22} {coef:+.3f}  {bar}")

    MODEL_OUT.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "models": all_weights,
        # legacy flat format for --predict (uses patronus weights as default)
        "weights": all_weights["patronus"]["weights"],
        "n_train": n,
        "loo_cv_mae": round(mae, 3),
    }, indent=2))
    print(f"\nSaved: {MODEL_OUT}")

    # Worst overall
    worst = sorted(zip(errors, ids, y, preds), reverse=True)[:5]
    print("\nWorst predictions (overall):")
    for err, vid, true, pred in worst:
        print(f"  {vid[:45]:<45} true={int(true)}  pred={pred:.1f}  err={err:.1f}")

    return all_weights["patronus"]["weights"]

# ─────────────────────────────────────────────────────────────────────────────
# Predict on new video
# ─────────────────────────────────────────────────────────────────────────────

def predict_video(mp4_path: str, template: str = None):
    """Score a single video. template: 'patronus' | 'pet_greeting' | None (auto-detect by path)."""
    try:
        import cv2, numpy as np
    except ImportError:
        print("cv2/numpy required for prediction")
        return

    if not MODEL_OUT.exists():
        print("No model found, training first...")
        train_and_eval()

    data = json.loads(MODEL_OUT.read_text())

    # Auto-detect template from path if not specified
    if template is None:
        p = mp4_path.lower()
        template = "patronus" if "patronus" in p else "pet_greeting"

    models = data.get("models", {})
    if template in models:
        w = models[template]["weights"]
        loo_mae = models[template]["loo_cv_mae"]
        n_train = models[template]["n_train"]
    else:
        w = data["weights"]
        loo_mae = data["loo_cv_mae"]
        n_train = data["n_train"]
        template = "default"
    print(f"Model: {template} (N={n_train}, LOO-MAE={loo_mae})")

    sys.path.insert(0, str(Path(__file__).parent))
    from optical_flow_label import compute_flow_magnitudes, detect_fluidity_defects

    mags = compute_flow_magnitudes(mp4_path)
    if len(mags) == 0:
        print("Could not read video")
        return

    result = detect_fluidity_defects(mags)
    defects = {k: v for k, v in result.items() if isinstance(v, bool)}

    feats = extract_features_from_flow(mags, defects)
    score = predict(w, feats)

    print(f"\nFluidity score: {score:.1f} / 10")
    print(f"Detected defects: {[k for k, v in defects.items() if v]}")
    print(f"Peak frame: ~{result.get('peak_frame_timestamp_s', '?')}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        predict_video(sys.argv[2])
    else:
        train_and_eval()
