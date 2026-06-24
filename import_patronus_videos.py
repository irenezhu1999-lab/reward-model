#!/usr/bin/env python3
"""
导入呼神护卫视频到 videos.jsonl + annotations.jsonl

运行：python import_patronus_videos.py
"""

import json
from pathlib import Path
from datetime import datetime

VIDEOS_FILE = Path(__file__).parent / "videos.jsonl"
ANNOTATIONS_FILE = Path(__file__).parent / "annotations.jsonl"
PATRONUS_DIR = Path("D:/AI Magic 效果/哈利波特魔法专项测试/呼神护卫")

# 加载现有记录
existing_videos = set()
if VIDEOS_FILE.exists():
    with open(VIDEOS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            existing_videos.add(rec.get("id") or rec.get("video_id"))

existing_annotations = set()
if ANNOTATIONS_FILE.exists():
    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            existing_annotations.add(rec.get("video_id"))

# 找所有 mp4
all_mp4s = list(PATRONUS_DIR.rglob("*.mp4"))
print(f"Found {len(all_mp4s)} mp4 files in {PATRONUS_DIR}")

# 导入
new_videos = []
new_annotations = []

for mp4_path in all_mp4s:
    rel_path = mp4_path.relative_to(PATRONUS_DIR.parent)
    video_id = f"patronus_{mp4_path.stem}"

    # 跳过已有的
    if video_id in existing_videos:
        continue

    # 创建 video 记录
    video_rec = {
        "id": video_id,  # 使用 "id" 字段以匹配现有格式
        "file_path": str(mp4_path),
        "template": "patronus",
        "version": "unknown",
        "subject_id": "unknown",
        "species": "human",
        "source_image": None,
        "prompt_version": "unknown",
        "model": "kling-v3-omni",
        "duration_s": 5.0,
        "resolution": "1080x1920",
        "gen_date": "2026-06-24",
        "meta": {},
    }
    new_videos.append(video_rec)

    # 创建空 annotation 记录（等光流标注填充）
    if video_id not in existing_annotations:
        ann_rec = {
            "annotation_id": f"ann_import_{video_id}",
            "video_id": video_id,
            "annotator": "auto_import",
            "annotation_date": "2026-06-24",
            "annotation_method": "pending_optical_flow",
            "scores": {
                "overall": None,
                "fluidity": None,
                "fidelity": None,
                "anatomy": None,
                "drama": None,
                "motion_correctness": None,
            },
            "defect_flags": {
                "motion_freeze": False,
                "uniform_velocity": False,
                "motion_stutter": False,
                "unnatural_trajectory": False,
                "joint_isolation": False,
                "identity_loss": False,
                "cross_species": False,
                "limb_deformation": False,
                "physics_violation": False,
                "lighting_incoherent": False,
                "fx_failure": False,
                "defocus_artifact": False,
                "incomplete_action": False,
            },
            "notes": "",
        }
        new_annotations.append(ann_rec)

# 追加到文件
with open(VIDEOS_FILE, "a", encoding="utf-8") as f:
    for rec in new_videos:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(ANNOTATIONS_FILE, "a", encoding="utf-8") as f:
    for rec in new_annotations:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Imported {len(new_videos)} new videos")
print(f"Created {len(new_annotations)} new annotation records")
print(f"Total videos: {len(existing_videos) + len(new_videos)}")
print(f"Total annotations: {len(existing_annotations) + len(new_annotations)}")
