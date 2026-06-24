# reward-model

AI 生成视频质量奖励模型 — 用于自动评分和筛选 AI 生成短视频。

## 背景

在批量生成 AI 视频（可灵 Keling / Wan2.1 等）时，人工逐条审查成本极高。本项目通过：
1. **光流自动标注** — 用 Farneback 光流检测 fluidity 维度的 4 类缺陷
2. **回归奖励模型** — 输入光流特征 + 缺陷 flag，输出 fluidity 分数 1-10
3. **结构化标注体系** — 13 维缺陷 flag，覆盖 5 个评分维度

目前已标注视频：219 条（宠物打招呼 80 条 + 呼神护卫 139 条）

---

## 目录结构

```
reward_model/
├── data_schema.md              # 标注格式规范 + 13维缺陷体系说明
├── build_dataset.py            # 构建训练集、特征提取
├── optical_flow_label.py       # 光流自动标注（Farneback）
├── train_fluidity_model.py     # Fluidity ridge regression 训练 + LOO-CV
├── import_patronus_videos.py   # 批量导入新视频目录
├── migrate_defects_v1_to_v2.py # 缺陷标记迁移工具（v1→v2）
├── annotations_public.jsonl    # 标注数据（脱敏，本地路径已移除）
└── videos_public.jsonl         # 视频元数据（脱敏）
```

---

## 评分维度

| 维度 | 说明 | 自动标注 |
|---|---|---|
| **fluidity** | 动作流畅度（加减速曲线、关节联动） | ✅ 光流检测 |
| **fidelity** | 人物/物种保真度 | 🔲 需人工 |
| **anatomy** | 解剖正确性（肢体形态） | 🔲 需人工 |
| **drama** | 视觉质感（光线/特效/构图） | 🔲 需人工 |
| **motion_correctness** | 动作完整性（是否完成预期动作） | 🔲 需人工 |

---

## 13 维缺陷 Flag

### fluidity
| Flag | 含义 |
|---|---|
| `motion_freeze` | 动作顶点或中途静止 ≥0.2s |
| `uniform_velocity` | 整段弧线无加减速曲线 |
| `motion_stutter` | 动作中途顿挫/不连贯 |
| `unnatural_trajectory` | 轨迹违反物理（急停/垂直弹跳） |
| `joint_isolation` | 单关节孤立运动，无联动 |

### fidelity
| Flag | 含义 |
|---|---|
| `identity_loss` | 人脸/物种特征偏离原图 |
| `cross_species` | 物种特征错误（爪变人手） |

### anatomy
| Flag | 含义 |
|---|---|
| `limb_deformation` | 肢体变形/多余/消失 |
| `physics_violation` | 违反物理规律 |

### drama
| Flag | 含义 |
|---|---|
| `lighting_incoherent` | 人物打光与场景不融合 |
| `fx_failure` | 特效失真（拉丝/穿模） |
| `defocus_artifact` | 尾帧明显虚焦/残影 |

### motion_correctness
| Flag | 含义 |
|---|---|
| `incomplete_action` | 动作未完成/道具消失 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install opencv-python numpy
```

### 2. 对新视频做光流标注

```bash
# 标注全部视频
python optical_flow_label.py

# 只处理特定模板
python optical_flow_label.py --filter "patronus"

# 预览不保存
python optical_flow_label.py --dry-run
```

### 3. 训练 Fluidity 模型

```bash
python train_fluidity_model.py
# 输出：fluidity_model_weights.json + LOO-CV 误差
```

### 4. 对单视频打分

```bash
python train_fluidity_model.py --predict /path/to/video.mp4
```

---

## 光流检测逻辑

`optical_flow_label.py` 使用 `cv2.calcOpticalFlowFarneback` 计算每帧间平均位移幅度，从幅度序列检测 4 类缺陷：

```
motion_freeze     — 运动高峰窗口内，连续帧幅度 < 峰值 10%
uniform_velocity  — 运动段 CV（变异系数）< 0.15（匀速机械感）
unnatural_trajectory — 末段幅度 < 峰值 5%（动作急停）
motion_stutter    — 运动段中途出现孤立低幅度帧
```

fluidity 分数从 8 分起，缺陷扣分，加速曲线加分，最终 clamp 到 1-10。

---

## 数据说明

`annotations_public.jsonl` 和 `videos_public.jsonl` 为脱敏版本：
- `file_path` 替换为 `<local_path>`
- `notes` 字段已清空
- video_id 保留（宠物系列用物种编号，无个人信息）

原始数据含本地绝对路径，不含个人隐私，仅因路径依赖本地环境而不提交。

---

## 关联项目

- [video-taste skill](https://github.com/irenezhu1999-lab/Claude-Code-skills-memory/tree/master/skills/video-taste) — 四旋钮 prompt 生成框架，FLUIDITY 旋钮与本项目评分体系对应
