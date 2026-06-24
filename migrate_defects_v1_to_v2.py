#!/usr/bin/env python3
"""
迁移脚本：将 annotations.jsonl 中的 defect_flags 从 v1 格式（19维）升级到 v2 格式（13维）

运行方式：
    python migrate_defects_v1_to_v2.py

会原地更新 annotations.jsonl，并生成备份文件 annotations.jsonl.v1_backup
"""

import json
from pathlib import Path
from build_dataset import DEFECT_MIGRATION_MAP, DEFECT_ALL

ANNOTATIONS_FILE = Path(__file__).parent / "annotations.jsonl"
BACKUP_FILE = ANNOTATIONS_FILE.with_suffix(".jsonl.v1_backup")

def migrate_defect_flags(old_flags: dict) -> dict:
    """将单条标注的 defect_flags 从 v1 迁移到 v2"""
    new_flags = {flag: False for flag in DEFECT_ALL}

    for old_flag, value in old_flags.items():
        if not value:
            continue  # 跳过 False 的 flag

        new_flag = DEFECT_MIGRATION_MAP.get(old_flag)

        if new_flag is None:
            # motion_stiff 删除，不迁移
            print(f"  Dropping {old_flag} (deleted in v2)")
            continue

        if new_flag not in DEFECT_ALL:
            print(f"  WARNING: {old_flag} maps to unknown flag {new_flag}")
            continue

        new_flags[new_flag] = True

    return new_flags

def main():
    if not ANNOTATIONS_FILE.exists():
        print(f"Error: {ANNOTATIONS_FILE} not found")
        return

    # 读取所有标注
    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        annotations = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(annotations)} annotations from {ANNOTATIONS_FILE}")

    # 备份
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")
    print(f"Backup saved to {BACKUP_FILE}")

    # 迁移
    migrated_count = 0
    for ann in annotations:
        if "defect_flags" not in ann or not ann["defect_flags"]:
            continue

        old_flags = ann["defect_flags"]
        new_flags = migrate_defect_flags(old_flags)

        # 如果有变化，更新
        if new_flags != old_flags:
            ann["defect_flags"] = new_flags
            migrated_count += 1
            print(f"Migrated video_id={ann['video_id']}")

    # 写回
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    print(f"\nMigration complete: {migrated_count}/{len(annotations)} records updated")
    print(f"v1 → v2: 19 defect types → 13 defect types")
    print(f"Updated file: {ANNOTATIONS_FILE}")

if __name__ == "__main__":
    main()
