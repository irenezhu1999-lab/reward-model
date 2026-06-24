#!/usr/bin/env python3
"""
用光流自动检测视频中的 FLUIDITY 缺陷：
- vertex_freeze: 顶点帧附近位移骤降（停顿）
- uniform_velocity: 连续N帧位移近似相等（匀速）
- abrupt_stop: 末段位移骤降为零
- mid_motion_stutter: 中途出现非预期静止帧

输出：更新 annotations.jsonl 里对应的 defect_flags 和 fluidity score
"""
import cv2
import numpy as np
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("D:/AI Magic 效果/reward_model")
ANNOTATIONS_JSONL = OUTPUT_DIR / "annotations.jsonl"
VIDEOS_JSONL = OUTPUT_DIR / "videos.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# 光流分析核心
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_magnitudes(video_path: str, max_frames: int = 200) -> np.ndarray:
    """返回每帧间的平均光流幅度 array，长度 = total_frames - 1"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.array([])

    magnitudes = []
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.array([])

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # downsample for speed
    h, w = prev_gray.shape
    scale = min(1.0, 320 / max(h, w))
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        prev_gray = cv2.resize(prev_gray, (new_w, new_h))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, (new_w, new_h))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        magnitudes.append(float(mag))
        prev_gray = gray
        frame_count += 1

    cap.release()
    return np.array(magnitudes)


def detect_fluidity_defects(magnitudes: np.ndarray, fps: float = 24.0) -> dict:
    """
    从光流幅度序列检测各类 FLUIDITY 缺陷。
    返回 defect dict + computed fluidity score。
    """
    if len(magnitudes) < 10:
        return {"error": "too_few_frames"}

    # 平滑 (去除单帧噪声)
    kernel = np.ones(3) / 3
    smooth = np.convolve(magnitudes, kernel, mode='same')

    # 找运动高峰段（幅度 > mean 的帧）
    mean_mag = smooth.mean()
    peak_threshold = mean_mag * 1.5
    motion_frames = np.where(smooth > peak_threshold)[0]

    defects = {}
    fluidity_score = 8  # start optimistic

    # ── 1. motion_freeze (v2: 合并 vertex_freeze + mid_motion_stutter) ─────────
    # 在运动高峰段内，找连续 N 帧位移 < 低阈值的片段
    # Only check freeze WITHIN the motion peak region (not the idle start/end)
    peak_idx = int(np.argmax(smooth))
    peak_mag = float(smooth.max())
    # Search window: ±15 frames around peak (0.6s at 24fps)
    window_start = max(0, peak_idx - 15)
    window_end = min(len(smooth), peak_idx + 15)
    peak_window = smooth[window_start:window_end]

    # Freeze = consecutive frames in peak window where magnitude < 10% of peak
    freeze_threshold = peak_mag * 0.10
    freeze_min_frames = max(3, int(fps * 0.15))  # ~0.15s

    in_freeze = False
    freeze_len = 0
    max_freeze = 0

    for m in peak_window:
        if m < freeze_threshold:
            if not in_freeze:
                in_freeze = True
                freeze_len = 1
            else:
                freeze_len += 1
            max_freeze = max(max_freeze, freeze_len)
        else:
            in_freeze = False
            freeze_len = 0

    motion_freeze = max_freeze >= freeze_min_frames
    defects["motion_freeze"] = bool(motion_freeze)
    if motion_freeze:
        fluidity_score -= 2

    # ── 2. uniform_velocity ───────────────────────────────────────────────────
    # 在运动段内，检查 std/mean 是否极低（匀速 → 变化小）
    if len(motion_frames) > 5:
        motion_mags = smooth[motion_frames]
        cv_coeff = motion_mags.std() / (motion_mags.mean() + 1e-6)  # coefficient of variation
        # 自然运动的 CV 通常 > 0.3；匀速机械运动 < 0.15
        uniform_velocity = cv_coeff < 0.15
        defects["uniform_velocity"] = bool(uniform_velocity)
        if uniform_velocity:
            fluidity_score -= 2
    else:
        defects["uniform_velocity"] = False

    # ── 3. unnatural_trajectory (v2: 合并 abrupt_stop + vertical_bounce) ───────
    # 检查末尾段：如果最后几帧幅度远小于运动段峰值
    last_n = max(3, int(fps * 0.2))
    tail = smooth[-last_n:]
    peak_mag = smooth.max()
    if peak_mag > 0:
        tail_ratio = tail.mean() / peak_mag
        unnatural_trajectory = tail_ratio < 0.05 and peak_mag > mean_mag * 1.5
    else:
        unnatural_trajectory = False
    defects["unnatural_trajectory"] = bool(unnatural_trajectory)
    if unnatural_trajectory:
        fluidity_score -= 1

    # ── 4. motion_stutter (v2: 重命名自 mid_motion_stutter) ─────────────────────
    # 在运动高峰段之间，找孤立的低幅度帧（不是开头结尾的静止）
    if len(motion_frames) > 0:
        first_peak = motion_frames[0]
        last_peak = motion_frames[-1]
        mid_segment = smooth[first_peak:last_peak]
        if len(mid_segment) > 5:
            stutter_threshold = mean_mag * 0.25
            stutter_frames = np.where(mid_segment < stutter_threshold)[0]
            motion_stutter = len(stutter_frames) >= 2
        else:
            motion_stutter = False
    else:
        motion_stutter = False
    defects["motion_stutter"] = bool(motion_stutter)
    if motion_stutter:
        fluidity_score -= 2

    # ── 5. Compute fluidity score (1-10) ──────────────────────────────────────
    # 正向指标：peak 加速曲线
    if len(motion_frames) > 4:
        # 找峰值帧
        peak_idx = np.argmax(smooth)
        # 启动段：从开始到峰值
        if peak_idx > 3:
            ramp_up = smooth[max(0, peak_idx-5):peak_idx]
            ramp_slope = np.polyfit(range(len(ramp_up)), ramp_up, 1)[0]
            if ramp_slope > 0.1:  # 有加速感
                fluidity_score = min(10, fluidity_score + 1)
        # 减速段：从峰值到结束
        if peak_idx < len(smooth) - 3:
            ramp_down = smooth[peak_idx:min(len(smooth), peak_idx+5)]
            ramp_slope = np.polyfit(range(len(ramp_down)), ramp_down, 1)[0]
            if ramp_slope < -0.1:  # 有减速感
                fluidity_score = min(10, fluidity_score + 1)

    fluidity_score = max(1, min(10, fluidity_score))

    # ── 6. Find peak frame index ──────────────────────────────────────────────
    peak_frame_idx = int(np.argmax(smooth)) + 1  # +1 because magnitudes[i] = diff between frame i and i+1
    peak_timestamp = peak_frame_idx / fps

    return {
        **defects,
        "fluidity_score": fluidity_score,
        "peak_frame_idx": peak_frame_idx,
        "peak_frame_timestamp_s": round(peak_timestamp, 2),
        "mean_flow_magnitude": round(float(mean_mag), 4),
        "peak_flow_magnitude": round(float(smooth.max()), 4),
        "motion_frame_count": int(len(motion_frames)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程：处理视频列表，更新 annotations
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run(video_filter: str = None, dry_run: bool = False):
    videos = load_jsonl(VIDEOS_JSONL)
    annotations = load_jsonl(ANNOTATIONS_JSONL)

    ann_by_vid = {a["video_id"]: a for a in annotations}

    updated = 0
    errors = 0
    for vid in videos:
        if video_filter and video_filter not in vid["id"]:
            continue

        fpath = vid["file_path"]
        # 尝试多种编码路径
        path_candidates = [fpath]
        if not Path(fpath).exists():
            # 尝试 UTF-8 re-encode
            try:
                path_candidates.append(fpath.encode('latin-1').decode('utf-8'))
            except:
                pass
            # 尝试直接从 id 推断（patronus 视频特殊处理）
            if 'patronus' in vid['id']:
                # patronus 视频都在 呼神护卫 目录下
                filename = vid['id'].replace('patronus_', '') + '.mp4'
                patronus_dir = Path("D:/AI Magic 效果/哈利波特魔法专项测试/呼神护卫")
                # 递归查找
                for p in patronus_dir.rglob(filename):
                    path_candidates.append(str(p))
                    break

        # 找第一个存在的路径
        actual_path = None
        for candidate in path_candidates:
            if Path(candidate).exists():
                actual_path = candidate
                break

        if not actual_path:
            print(f"  [SKIP] not found: {fpath}")
            continue

        fpath = actual_path

        print(f"Processing {vid['id']} ...", end=" ", flush=True)
        mags = compute_flow_magnitudes(fpath)
        if len(mags) == 0:
            print("ERROR: no frames")
            errors += 1
            continue

        fps = 24.0  # default; could read from video metadata
        result = detect_fluidity_defects(mags, fps=fps)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            errors += 1
            continue

        fluidity_score = result.pop("fluidity_score")
        peak_frame_idx = result.pop("peak_frame_idx")
        peak_ts = result.pop("peak_frame_timestamp_s")
        mean_flow = result.pop("mean_flow_magnitude")
        peak_flow = result.pop("peak_flow_magnitude")
        motion_count = result.pop("motion_frame_count")

        auto_defects = result  # remaining keys are defect flags

        print(f"fluidity={fluidity_score}, peak@{peak_ts}s, defects={[k for k,v in auto_defects.items() if v]}")

        if not dry_run:
            ann = ann_by_vid.get(vid["id"])
            if ann:
                # merge: auto labels only for flags that weren't already manually set
                for flag, val in auto_defects.items():
                    # don't overwrite anatomy flags (optical flow can't detect anatomy)
                    if flag not in ("cross_species", "limb_deformation"):
                        ann["defect_flags"][flag] = val
                if ann["scores"]["fluidity"] is None:
                    ann["scores"]["fluidity"] = fluidity_score
                ann["peak_frame_idx"] = peak_frame_idx
                ann["peak_frame_timestamp_s"] = peak_ts
                ann["sampling_method"] = "optical_flow_farneback"
                ann["notes"] = (ann.get("notes") or "") + f" [auto_flow: mean={mean_flow:.3f} peak={peak_flow:.3f} motion_frames={motion_count}]"
            else:
                # create new annotation record (shouldn't happen if import script ran)
                from build_dataset import DEFECT_ALL
                ann = {
                    "annotation_id": f"ann_flow_{vid['id']}",
                    "video_id": vid["id"],
                    "annotator": "optical_flow_auto",
                    "annotation_date": "2026-06-24",
                    "annotation_method": "optical_flow_farneback",
                    "scores": {
                        "overall": None,
                        "fluidity": fluidity_score,
                        "fidelity": None,
                        "anatomy": None,
                        "drama": None,
                        "motion_correctness": None,
                    },
                    "defect_flags": {flag: auto_defects.get(flag, False) for flag in DEFECT_ALL},
                    "peak_frame_idx": peak_frame_idx,
                    "peak_frame_timestamp_s": peak_ts,
                    "sampling_method": "optical_flow_farneback",
                    "notes": f"[auto_flow: mean={mean_flow:.3f} peak={peak_flow:.3f} motion_frames={motion_count}]",
                }
                annotations.append(ann)
                ann_by_vid[vid["id"]] = ann

        updated += 1

    if not dry_run:
        save_jsonl(ANNOTATIONS_JSONL, annotations)
        print(f"\nUpdated {updated} annotations, {errors} errors.")
        print(f"Saved to {ANNOTATIONS_JSONL}")
    else:
        print(f"\n[DRY RUN] Would update {updated} annotations, {errors} errors.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None, help="Only process videos whose id contains this string")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just print")
    args = parser.parse_args()
    run(video_filter=args.filter, dry_run=args.dry_run)
