#!/usr/bin/env python3
"""从现有 QA 报告和目录结构批量构建奖励模型数据集"""
import json
import csv
import re
import hashlib
from pathlib import Path
from datetime import datetime

BASE = Path("D:/AI Magic 效果")
OUTPUT_DIR = Path("D:/AI Magic 效果/reward_model")
VIDEOS_JSONL = OUTPUT_DIR / "videos.jsonl"
ANNOTATIONS_JSONL = OUTPUT_DIR / "annotations.jsonl"
COMPARISONS_JSONL = OUTPUT_DIR / "comparisons.jsonl"

def make_id(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"[^\w]", "_", stem)[:80]

def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Source 1: 宠物系列 v1 / v2
# ─────────────────────────────────────────────────────────────────────────────

PET_SPECIES = {
    "01_golden_retriever": "dog",
    "02_labrador": "dog",
    "03_tabby_cat": "cat",
    "04_persian_cat": "cat",
    "05_hamster": "hamster",
    "06_rabbit_white": "rabbit",
    "07_rabbit_brown": "rabbit",
    "08_parakeet": "bird",
    "09_cockatiel": "bird",
    "10_guinea_pig": "rodent",
}

# Gemini QA scores from gemini_qa_report_v2.txt (manually extracted)
PET_GEMINI_SCORES = {
    "v1": {
        "01_golden_retriever": {"overall": 7, "defects": []},
        "02_labrador":         {"overall": 5, "defects": ["defocus_artifact"]},
        "03_tabby_cat":        {"overall": 7, "defects": []},
        "04_persian_cat":      {"overall": 1, "defects": ["human_hand"]},
        "05_hamster":          {"overall": 5, "defects": ["defocus_artifact"]},
        "06_rabbit_white":     {"overall": 6, "defects": []},
        "07_rabbit_brown":     {"overall": 6, "defects": ["defocus_artifact"]},
        "08_parakeet":         {"overall": 6, "defects": []},
        "09_cockatiel":        {"overall": 6, "defects": []},
        "10_guinea_pig":       {"overall": 5, "defects": []},
    },
    "v2": {
        "01_golden_retriever": {"overall": 7, "defects": []},
        "02_labrador":         {"overall": 8, "defects": []},
        "03_tabby_cat":        {"overall": 7, "defects": []},
        "04_persian_cat":      {"overall": 9, "defects": []},
        "05_hamster":          {"overall": 8, "defects": []},
        "06_rabbit_white":     {"overall": 7, "defects": []},
        "07_rabbit_brown":     {"overall": 7, "defects": []},
        "08_parakeet":         {"overall": 7, "defects": []},
        "09_cockatiel":        {"overall": 6, "defects": []},
        "10_guinea_pig":       {"overall": 6, "defects": []},
    },
    "v3": {
        "01_golden_retriever": {"overall": 7, "defects": ["vertex_freeze"]},
        "04_persian_cat":      {"overall": 6, "defects": ["vertex_freeze", "uniform_velocity"]},
        "08_parakeet":         {"overall": 7, "defects": []},
    },
}

DEFECT_ALL = [
    # fluidity（动作流畅度）— 光流自动 + 人工
    "motion_freeze", "uniform_velocity", "motion_stutter",
    "unnatural_trajectory", "joint_isolation",
    # fidelity（人物/物种保真度）— ArcFace自动 + 人工
    "identity_loss", "cross_species",
    # anatomy（解剖正确性）— 人工
    "limb_deformation", "physics_violation",
    # drama（视觉质感）— 人工 + 光流
    "lighting_incoherent", "fx_failure", "defocus_artifact",
    # motion_correctness（动作完整性）— 人工
    "incomplete_action",
]

# v1 → v2 映射表（用于迁移现有标注）
DEFECT_MIGRATION_MAP = {
    "vertex_freeze": "motion_freeze",
    "mid_motion_stutter": "motion_freeze",
    "uniform_velocity": "uniform_velocity",
    "abrupt_stop": "unnatural_trajectory",
    "vertical_bounce": "unnatural_trajectory",
    "isolated_joint": "joint_isolation",
    "face_turn": "identity_loss",
    "identity_drift": "identity_loss",
    "human_hand": "cross_species",
    "hand_distortion": "limb_deformation",
    "extra_limbs": "limb_deformation",
    "physics_error": "physics_violation",
    "lighting_mismatch": "lighting_incoherent",
    "fx_artifact": "fx_failure",
    "penetration_artifact": "fx_failure",
    "defocus_artifact": "defocus_artifact",
    "early_stop": "incomplete_action",
    "prop_disappear": "incomplete_action",
    "motion_stiff": None,  # 删除，不迁移
}

def build_defect_flags(defects: list[str]) -> dict:
    return {d: d in defects for d in DEFECT_ALL}

def collect_pet_records():
    videos, annotations = [], []
    version_dirs = {
        "v1": BASE / "宠物系列/videos_greeting",
        "v2": BASE / "宠物系列/videos_greeting_v2",
        "v3": BASE / "宠物系列/videos_greeting_v3_test",
    }
    version_prompts = {"v1": "v1_original_chinese", "v2": "v2_english_anatomic", "v3": "v3_fluidity_aware"}

    for ver, vdir in version_dirs.items():
        if not vdir.exists():
            continue
        for mp4 in vdir.glob("*.mp4"):
            if "preview" in mp4.stem:
                continue
            subject_id = re.match(r"(\d+_\w+?)(?:_v\d+)?(?:_\d+)?$", mp4.stem)
            if not subject_id:
                continue
            sid = subject_id.group(1)

            vid_id = make_id(mp4.name)
            rec = {
                "id": vid_id,
                "file_path": str(mp4),
                "template": "pet_greeting",
                "version": ver,
                "subject_id": sid,
                "species": PET_SPECIES.get(sid, "unknown"),
                "source_image": str(BASE / f"宠物系列/{sid}.jpg"),
                "prompt_version": version_prompts[ver],
                "model": "kling-v3-omni",
                "duration_s": 5.0,
                "resolution": "16:9",
                "gen_date": "2026-06-18",
                "meta": {
                    "target_platform": "douyin",
                    "target_dials": {"drama": 4, "motion": 3, "fidelity": 8, "fluidity": 8}
                }
            }
            videos.append(rec)

            qa = PET_GEMINI_SCORES.get(ver, {}).get(sid)
            if qa:
                ann = {
                    "annotation_id": f"ann_gemini_{vid_id}",
                    "video_id": vid_id,
                    "annotator": "gemini-2.5-flash" if ver in ("v1", "v2") else "claude-sonnet-4-6",
                    "annotation_date": "2026-06-18" if ver != "v3" else "2026-06-23",
                    "annotation_method": "keyframe_grid" if ver != "v3" else "consecutive_5frame_peak",
                    "scores": {
                        "overall": qa["overall"],
                        "fluidity": None,
                        "fidelity": None,
                        "anatomy": None,
                        "drama": None,
                        "motion_correctness": None,
                    },
                    "defect_flags": build_defect_flags(qa["defects"]),
                    "peak_frame_idx": 90 if ver == "v3" and sid == "04_persian_cat" else None,
                    "peak_frame_timestamp_s": 3.75 if ver == "v3" and sid == "04_persian_cat" else None,
                    "sampling_method": "consecutive_5frame_peak" if ver == "v3" else "sparse_keyframe",
                    "notes": "",
                }
                # fill known fluidity scores from v3 manual audit
                if ver == "v3" and sid == "04_persian_cat":
                    ann["scores"]["fluidity"] = 5
                    ann["notes"] = "顶点 freeze 确认：帧 87-91 爪位置几乎不变。挥手周期未完成。"
                annotations.append(ann)

    return videos, annotations

# ─────────────────────────────────────────────────────────────────────────────
# Source 2: 情人节黑夜八音盒 binary labels
# ─────────────────────────────────────────────────────────────────────────────

VALENTINES_DIR = BASE / "情人节训练数据/tt 素材/黑夜八音盒（标注）"
DEFECT_DIR_MAP = {
    "可用": [],
    "不可用": [],
    "烟花问题": ["fx_artifact"],
    "腿问题（不明显）": ["anatomy"],
}

def collect_valentines_records():
    videos, annotations = [], []
    if not VALENTINES_DIR.exists():
        return videos, annotations

    for subdir, defects in DEFECT_DIR_MAP.items():
        d = VALENTINES_DIR / subdir
        if not d.exists():
            continue
        for mp4 in d.glob("*.mp4"):
            vid_id = make_id(mp4.name) + "_val"
            usable = subdir == "可用"
            overall = 8 if usable else (5 if defects else 2)

            rec = {
                "id": vid_id,
                "file_path": str(mp4),
                "template": "valentines_music_box",
                "version": "v1",
                "subject_id": mp4.stem,
                "species": None,
                "source_image": None,
                "prompt_version": "valentines_v1",
                "model": "unknown",
                "duration_s": None,
                "resolution": "9:16",
                "gen_date": "2026-01-22",
                "meta": {"target_platform": "douyin", "target_dials": None}
            }
            videos.append(rec)

            all_defects = defects.copy()
            ann_defect_flags = build_defect_flags(all_defects)
            ann = {
                "annotation_id": f"ann_human_{vid_id}",
                "video_id": vid_id,
                "annotator": "human",
                "annotation_date": "2026-01-22",
                "annotation_method": "manual_folder_sort",
                "scores": {
                    "overall": overall,
                    "fluidity": None,
                    "fidelity": None,
                    "anatomy": None,
                    "drama": None,
                    "motion_correctness": None,
                },
                "defect_flags": ann_defect_flags,
                "peak_frame_idx": None,
                "peak_frame_timestamp_s": None,
                "sampling_method": "manual_playback",
                "notes": f"folder: {subdir}",
            }
            annotations.append(ann)

    return videos, annotations

# ─────────────────────────────────────────────────────────────────────────────
# Source 3: 自动生成宠物对比 pairs（v1 vs v2，同 subject）
# ─────────────────────────────────────────────────────────────────────────────

def build_pet_comparisons(videos: list[dict]) -> list[dict]:
    by_subject: dict[str, dict[str, str]] = {}
    for v in videos:
        if v["template"] == "pet_greeting":
            sid = v["subject_id"]
            ver = v["version"]
            by_subject.setdefault(sid, {})[ver] = v["id"]

    comparisons = []
    for sid, ver_map in by_subject.items():
        v1_id = ver_map.get("v1")
        v2_id = ver_map.get("v2")
        v3_id = ver_map.get("v3")
        if v1_id and v2_id:
            s1 = PET_GEMINI_SCORES["v1"].get(sid, {}).get("overall", 5)
            s2 = PET_GEMINI_SCORES["v2"].get(sid, {}).get("overall", 5)
            winner = "a" if s2 >= s1 else "b"  # a=v2, b=v1 by convention
            comparisons.append({
                "comparison_id": f"cmp_pet_{sid}_v2_vs_v1",
                "video_a": v2_id,
                "video_b": v1_id,
                "winner": winner,
                "confidence": "high" if abs(s2 - s1) >= 2 else "medium",
                "annotator": "gemini-2.5-flash",
                "annotation_date": "2026-06-18",
                "dimension": "overall",
                "notes": f"v2 score={s2}, v1 score={s1}",
            })
        if v2_id and v3_id:
            s2 = PET_GEMINI_SCORES["v2"].get(sid, {}).get("overall", 5)
            s3 = PET_GEMINI_SCORES["v3"].get(sid, {}).get("overall", 5)
            winner = "a" if s3 >= s2 else "b"
            comparisons.append({
                "comparison_id": f"cmp_pet_{sid}_v3_vs_v2",
                "video_a": v3_id,
                "video_b": v2_id,
                "winner": winner,
                "confidence": "medium",
                "annotator": "claude-sonnet-4-6",
                "annotation_date": "2026-06-23",
                "dimension": "fluidity",
                "notes": f"v3 score={s3}, v2 score={s2}",
            })
    return comparisons

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_videos, all_annotations = [], []

    pet_v, pet_a = collect_pet_records()
    all_videos += pet_v
    all_annotations += pet_a
    print(f"Pet series: {len(pet_v)} videos, {len(pet_a)} annotations")

    val_v, val_a = collect_valentines_records()
    all_videos += val_v
    all_annotations += val_a
    print(f"Valentines: {len(val_v)} videos, {len(val_a)} annotations")

    comparisons = build_pet_comparisons(all_videos)
    print(f"Comparisons auto-generated: {len(comparisons)}")

    write_jsonl(VIDEOS_JSONL, all_videos)
    write_jsonl(ANNOTATIONS_JSONL, all_annotations)
    write_jsonl(COMPARISONS_JSONL, comparisons)

    print("\nDataset summary:")
    print(f"  Total videos:      {len(all_videos)}")
    print(f"  Total annotations: {len(all_annotations)}")
    print(f"  Total comparisons: {len(comparisons)}")

if __name__ == "__main__":
    main()
