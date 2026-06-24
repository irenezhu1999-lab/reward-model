# 奖励模型数据格式设计 v0.1

## 背景

目标：训练一个能自动评分 AI 生成视频质量的奖励模型，用于筛选/排序生成结果。
现有数据：~1200 个 mp4，跨宠物/哈利波特/Alicorn/情人节等模板。
标注现状：情人节系列有 50 条二分类标注（可用/不可用）；宠物系列有 Gemini QA 评分。

---

## 数据格式：两层 JSONL

### Layer 1：视频元数据条目（video_record）

```json
{
  "id": "pet_v2_04_persian_cat_84757821",
  "file_path": "D:/AI Magic 效果/宠物系列/videos_greeting_v2/04_persian_cat_v2_84757821.mp4",
  "template": "pet_greeting",
  "version": "v2",
  "subject_id": "04_persian_cat",
  "species": "cat",
  "source_image": "D:/AI Magic 效果/宠物系列/04_persian_cat.jpg",
  "prompt_version": "v2",
  "model": "kling-v3-omni",
  "duration_s": 5.0,
  "resolution": "16:9",
  "gen_date": "2026-06-18",
  "meta": {
    "target_platform": "douyin",
    "target_dials": {"drama": 4, "motion": 3, "fidelity": 8, "fluidity": 8}
  }
}
```

---

### Layer 2：标注条目（annotation_record）

每条视频可有多个标注（不同标注者、不同轮次），取均值或最高信任度标注作为 label。

#### 2A. 绝对评分格式（用于 regression RM）

```json
{
  "annotation_id": "ann_20260623_001",
  "video_id": "pet_v2_04_persian_cat_84757821",
  "annotator": "gemini-2.5-flash",
  "annotation_date": "2026-06-18",
  "annotation_method": "keyframe_grid",

  "scores": {
    "overall": 9,
    "fluidity": 6,
    "fidelity": 9,
    "anatomy": 8,
    "drama": 4,
    "motion_correctness": 7
  },

  "defect_flags": {
    "human_hand": false,
    "vertex_freeze": true,
    "uniform_velocity": true,
    "isolated_joint": false,
    "abrupt_stop": false,
    "mid_motion_stutter": false,
    "vertical_bounce": false,
    "extra_limbs": false,
    "defocus_artifact": false
  },

  "peak_frame_idx": 90,
  "peak_frame_timestamp_s": 3.75,
  "sampling_method": "ffprobe_all_frames + peak_scan",

  "notes": "顶点 freeze：帧 87-91 爪位置几乎不变，约 0.2s 静止。挥手周期未完成。"
}
```

#### 2B. 两两比较格式（用于 pairwise RM，标注成本更低）

```json
{
  "comparison_id": "cmp_20260623_001",
  "video_a": "pet_v2_04_persian_cat_84757821",
  "video_b": "pet_v1_04_persian_cat_85432872",
  "winner": "a",
  "confidence": "high",
  "annotator": "human",
  "annotation_date": "2026-06-23",
  "dimension": "fluidity",
  "notes": "v2 修复了人手问题但有顶点 freeze；v1 完全是人手。v2 仍更好。"
}
```

---

## 标注维度定义

| 维度 | 含义 | 打分依据 |
|------|------|---------|
| `overall` | 综合质量，发布可用性 | 1-10，参照现有 Gemini 评分 |
| `fluidity` | 动作流畅度 | §1.C 5帧法：加速曲线/关节联动/ending缓冲 |
| `fidelity` | 物种/品种形态保真 | 爪型、毛色、体型是否正确 |
| `anatomy` | 解剖正确性 | 趾数、肉垫形态、无多肢/人手 |
| `drama` | 构图/光线质感 | 是否达到旋钮目标值 |
| `motion_correctness` | 动作完整性 | 挥手是否完成完整周期 |

---

## 缺陷标记（defect_flags）定义

**设计原则：** 按评分维度组织，不按模板分；每个维度下只保留必要且不重叠的缺陷；优先保留可量化检测的特征。

### fluidity（动作流畅度）

| 标记 | 触发条件 | 检测方式 | 扣分 |
|------|---------|---------|------|
| `motion_freeze` | 动作顶点或中途出现 ≥0.2s 静止 | 光流自动 | -2 |
| `uniform_velocity` | 匀速运动，无加减速曲线（CV < 0.15） | 光流自动 | -2 |
| `motion_stutter` | 动作中途顿挫/不连贯 | 光流自动 | -2 |
| `unnatural_trajectory` | 轨迹违反物理（垂直弹跳/突然方向变化/急停） | 光流自动 | -1 |
| `joint_isolation` | 单关节孤立运动，肩背无联动 | 人工标注 | -1 |

### fidelity（人物/物种保真度）

| 标记 | 触发条件 | 检测方式 | 扣分 |
|------|---------|---------|------|
| `identity_loss` | 人脸/物种特征不像原图（ArcFace < 0.7 或品种错误） | ArcFace自动 | -3 |
| `cross_species` | 物种特征错误（爪变人手、狗变猫） | 人工标注 | -3 |

### anatomy（解剖正确性）

| 标记 | 触发条件 | 检测方式 | 扣分 |
|------|---------|---------|------|
| `limb_deformation` | 肢体变形、多余肢体、肢体消失 | 人工标注 | -2 |
| `physics_violation` | 违反物理（人物浮空、守护神走路不飘） | 人工标注 | -1 |

### drama（视觉质感）

| 标记 | 触发条件 | 检测方式 | 扣分 |
|------|---------|---------|------|
| `lighting_incoherent` | 人物打光与场景不融合 | 人工标注 | -1 |
| `fx_failure` | 特效失真（拉丝/放射状/穿模） | 人工标注 | -2 |
| `defocus_artifact` | 尾帧虚焦/残影 | 光流自动 | -1 |

### motion_correctness（动作完整性）

| 标记 | 触发条件 | 检测方式 | 扣分 |
|------|---------|---------|------|
| `incomplete_action` | 动作未完成、道具消失、ending构图乱 | 人工标注 | -2 |

---

## 旧版 flag 映射表（v1 → v2 迁移用）

| 旧 flag（v1, 19个） | 新 flag（v2, 13个） | 备注 |
|---|---|---|
| `vertex_freeze` / `mid_motion_stutter` | `motion_freeze` | 合并：都是不该停的时候停了 |
| `uniform_velocity` | `uniform_velocity` | 保留 |
| `abrupt_stop` / `vertical_bounce` | `unnatural_trajectory` | 合并：都是轨迹异常 |
| `isolated_joint` | `joint_isolation` | 重命名 |
| `face_turn` / `identity_drift` | `identity_loss` | 合并：侧脸是原因，不像是结果 |
| `human_hand` | `cross_species` | 归入更通用的物种错误 |
| `hand_distortion` / `extra_limbs` | `limb_deformation` | 合并 |
| `physics_error` | `physics_violation` | 重命名 |
| `lighting_mismatch` | `lighting_incoherent` | 重命名 |
| `fx_artifact` / `penetration_artifact` | `fx_failure` | 合并：都是魔法效果问题 |
| `defocus_artifact` | `defocus_artifact` | 保留 |
| `early_stop` / `prop_disappear` | `incomplete_action` | 合并：该有的东西没了 |
| `motion_stiff` | ❌ 删除 | 用其他 fluidity flag 组合判断 |

---

## 文件结构

```
D:/AI Magic 效果/reward_model/
├── data_schema.md          ← 本文件
├── videos.jsonl            ← 所有视频元数据（一行一条）
├── annotations.jsonl       ← 绝对评分标注（一行一条）
├── comparisons.jsonl       ← 两两比较标注（一行一条）
├── build_dataset.py        ← 从已有 QA 报告批量导入标注
└── README.md
```

---

## 现有数据导入映射

| 已有标注来源 | 映射到 | 字段映射 |
|------------|--------|---------|
| `gemini_qa_report_v2.txt` 每行评分 | `annotations.jsonl` | score → `overall`，issues → `defect_flags` |
| `final_verdict.txt` 表格 | `annotations.jsonl` | FIXED/NEW_ISSUE → `defect_flags` |
| `annotated_1769063981800.csv` 可用/不可用 | `annotations.jsonl` | 可用=8，不可用=2（`overall` 粗标） |
| `黑夜八音盒（标注）/` 目录分类 | `annotations.jsonl` | 子目录名 → `defect_flags`（烟花问题→specific flag） |
| v3 审查结论（本会话）| `annotations.jsonl` | 04_persian_cat FLUIDITY 5/10，vertex_freeze=true |

---

## 训练规模建议

| 当前阶段 | 标注数量目标 | 建议模型 |
|---------|------------|---------|
| 冷启动 | 50-100 条绝对评分 | GBDT on 光流特征 |
| 扩展期 | 200+ 条 | VideoMAE fine-tune + 回归头 |
| 两两比较 | 现有 50 条 × 配对 ≈ 300+ 对 | Bradley-Terry RM |

宠物系列 v1/v2 共 29 条视频 + 情人节 50 条 = **79 条可直接导入的标注**，能跑冷启动。

---

## 当前数据集缺口分析（2026-06-23 运行结果）

运行 `build_dataset.py` 后：80 条视频，80 条标注，13 条比较对。

**问题 1：类别严重不平衡**
- score=2（情人节"不可用"粗标）占 39/80 = **49%**
- 情人节二分类标注质量低：只有可用/不可用，无缺陷原因，建议重标或单独训分类头

**问题 2：fluidity 细分标注极少**
- 80 条里只有 1 条有 fluidity 分（04_persian_cat v3）
- 下一步：对宠物系列全部用 5帧法补标 fluidity（约 2小时工作量）

**问题 3：defect 正例严重不足**
| 缺陷类型 | 正例数 | 建议 |
|---------|--------|------|
| human_hand | 1 | 已有 v1 persian_cat；找更多例子 |
| vertex_freeze | 2 | 宠物 v3 有 2 个；可程序化检测扩增 |
| uniform_velocity | 1 | 同上 |
| isolated_joint | 0 | 需专门标注 |
| abrupt_stop | 0 | 需专门标注 |
| vertical_bounce | 0 | 需专门标注 |

**下一步优先级：**
1. 用光流脚本自动检测 `vertex_freeze` / `uniform_velocity`，批量标注哈利波特系列
2. 对宠物 v1/v2 全 29 条补标 fluidity（人工 or Gemini）
3. 情人节 39 条"不可用"重标缺陷原因（`vertical_bounce`/`anatomy` 等）
