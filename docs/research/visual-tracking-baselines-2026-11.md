# 足球直播转竖屏：视觉检测与跟踪基线调研

核验日期：**2026-11-14**。范围：2024—2026 年模型、作者论文和官方仓库；保留 ByteTrack 等较早方法作为廉价工程参照。本次没有安装、下载或运行模型，所有足球适用性判断均是待验证的设计推断，不能据此宣称实际效果或成本胜出。

## 结论与建议

第一轮视觉基线建议用 **YOLO26s 检测 + 显式配置 BoT-SORT + 固定规则决定关注区域 + 共同裁切控制器**。同时保留一个更简单的 ByteTrack 变体；SAM 3.1 作为第二阶段较强视觉基线。SAMURAI、SAM2Long 用于定位“遮挡后恢复是否是主要瓶颈”，无需第一轮全接入。

本项目的目标已经从“跟住指定人物”变成“自动决定足球事件中应该看哪里”。检测框、对象 ID 和分割掩码只是观察结果。传球后关注接球人、射门时兼顾球门、进球后关注庆祝者，仍然需要导演策略。因此不能把一个有人工首帧目标的 SAM 跟踪器，与无人工指定的 Drex agent 直接比较为同一个任务。

## 已核验候选

| 组件 | 时间与真实能力 | 输入 → 输出 | 在本项目中的位置 |
|---|---|---|---|
| SAM 2 / 2.1 | SAM 2 为 2024 年；2.1 检查点为 2024 年 9 月。跨帧 streaming memory，支持多目标与后续交互修正 | 视频 + 点/框/掩码提示 → 带对象 ID 的逐帧 mask | 测量指定目标的定位与遮挡恢复；本身没有文本语义导演能力 |
| SAM 3 | 论文首次提交 2025-11-20；概念提示分割，检测器与记忆跟踪器共享 backbone | 视频/图像 + 短文本、图像范例或视觉提示 → 匹配概念的多个对象 mask、身份；图像接口另给 boxes/scores | 无人工点选情况下，用“soccer ball”“players”等概念产生观察结果 |
| SAM 3.1 | 官方发布 2026-03-27；Object Multiplex 将多个对象分组共享处理/记忆 | 延续 SAM 3 的提示式视频分割接口，提供文本与点提示示例 | 较强开放词汇视觉 baseline；多对象规模较大时优化更有意义 |
| SAMURAI | 2024-11 论文；基于 SAM 2 的免训练运动感知记忆选择 | 官方自定义视频 demo：视频 + 首帧 bbox 文本文件 → 跟踪结果；底层使用 mask | 固定球员或球的短镜头跟踪诊断，不能自动选择精彩事件主体 |
| SAM2Long | 2024 年预印本、ICCV 2025；维护多个分割路径的免训练 memory tree | 初始对象提示 + 视频 → 整段分割路径 | 离线遮挡/重现对照；原版最终输出不能直接称作零前瞻直播结果 |
| YOLO26 + MOT | YOLO26 于 2026 年 1 月发布；ByteTrack、BoT-SORT 为 2022 年工程方法 | 帧 → 人/球框、类别和置信度；关联器增加 track ID | 最易控制成本和恢复频率的视觉参照 |

SAM 系列依据：[SAM 2 官方仓库](https://github.com/facebookresearch/sam2)、[SAM 2 视频接口源码](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py)、[SAM 3 论文](https://arxiv.org/abs/2511.16719)、[SAM 3 官方仓库](https://github.com/facebookresearch/sam3)、[SAM 3.1 发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)。SAM 2 README 对 2.1 分别出现 9 月 29 日与 9 月 30 日，故这里只固定到月份。

跟踪增强依据：[SAMURAI 论文](https://arxiv.org/abs/2411.11922)、[SAMURAI 官方仓库](https://github.com/yangchris11/samurai)、[SAM2Long 论文](https://arxiv.org/html/2410.16268v3)、[SAM2Long 官方仓库](https://github.com/Mark12Ding/SAM2Long)。检测与关联依据：[YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26)、[Ultralytics 跟踪文档](https://docs.ultralytics.com/modes/track)、[ByteTrack 官方仓库](https://github.com/FoundationVision/ByteTrack)。

## 不能直接移植的论文结论

SAM 3.1 宣称的约 7 倍加速，对应 **单张 H100、128 个对象、相对 2025 年 11 月版 SAM 3**；不能换算成本项目仅追球或几个人时的加速。官方表中视频概念分割的不同数据集结果有升有降，因此也不能说 3.1 所有能力全面提高。其评测对象是分割/概念跟踪，而非足球竖屏观感。[官方发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

SAMURAI 的 LaSOT、GOT-10k 等结果支持它作为单目标跟踪候选；没有证明远景几像素足球、频繁转播切镜和精彩事件主体选择已经解决。官方 demo 需要首帧框，其输入是 `x,y,w,h`，SAM 2 内部框接口则使用 `x1,y1,x2,y2`；统一 adapter 时应明确转换。[论文](https://arxiv.org/abs/2411.11922)、[官方 README](https://github.com/yangchris11/samurai)

SAM2Long 每帧扩展与剪枝候选路径，但论文明确在最后一帧处理完成后选择累计分数最高的整条路径。其官方完整结果应列为**离线视觉恢复对照**。若改成每帧提交当前最优结果，需标成我们实现的 causal 变体，并重新评测，不能继承原论文指标。[方法说明](https://arxiv.org/html/2410.16268v3)

YOLO26 的 COCO 精度与特定设备延迟不能证明本地足球球检测可靠，也不能表示完整视频处理速度。先用现有 30 帧检查人/球召回，尤其检查原图缩放后的小球；如果 pretrained sports-ball 类明显失效，应增加足球专用检测器这一明确变体，而非悄悄靠手工框把结果补好。[模型说明](https://docs.ultralytics.com/models/yolo26)

## 足球专项的更相关参照

**SoccerNet Game State Reconstruction，CVPRW 2024**：从移动转播机位检测、跟踪并识别球员，重建球场小地图；提供 200 个标注片段。当前官方开发包包含 YOLOv11、PRTReID/BPBreID、球场标定和号码识别等模块，可借鉴足球身份/队伍信息和 tracker state 的管理方式。它不是现成竖屏导演，完整接入会引入当前不需要的标定任务。[官方仓库](https://github.com/SoccerNet/sn-gamestate)、[论文](https://arxiv.org/abs/2404.11335)

**From Broadcast to Minimap，CVPRW 2025**：作者报告获得 SoccerNet GSR 2024 第一名。其方案仍使用足球微调 YOLOv5、ReID、队伍与号码信息以及后处理，说明“新模型年份”不是足球端到端效果的可靠替代指标。其离线合并轨迹阶段也不能自动归入直播比较。这是领域工程参照，不建议第一轮复现整套。[作者论文](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Golovkin_From_Broadcast_to_Minimap_Achieving_State-of-the-Art_SoccerNet_Game_State_Reconstruction_CVPRW_2025_paper.pdf)

## 最小视觉 baseline 的具体规格

以下均为本项目建议，参数需要在开发片段固定，之后不看测试结果调整。

1. **输入**：共同的连续片段、时间戳和镜头边界。第一轮使用相同边界列表，让切镜检测质量暂时不成为混杂因素。
2. **观察器**：YOLO26s；固定模型权重、输入分辨率、置信度阈值和检测 cadence。每帧检测作为完整计算参照，再单独跑固定降采样频率的成本变体。不要把固定低频基线与全帧模型比较时只报调用次数。
3. **跟踪记忆**：球与球员分开管理，保存最近框、速度、可见性和失踪时长。球员用 BoT-SORT；球可用同一框关联方法起步，但快速飞球导致的运动模型失败必须单独记录，不能因为 tracker ID 未变化就判定成功。
4. **规则导演**：可信球位置优先；球附近球员形成候选动作区域；球短暂不可见时有限时保持/预测；长期不可见时明确进入安全回退。特写时用主要人物区域。所有规则写死，不用 Drex/Gemini 暗中替视觉基线决定关注对象。
5. **共同控制器**：所有方法将目标区域交给同一 9:16 crop planner；共同的死区、速度/加速度限制和镜头内平滑。源视频真正切镜时允许 reset，不能跨切镜强行平滑。离线前后双向平滑与直播单向滤波分开报告。
6. **输出与日志**：逐帧 crop、目标/可见性、恢复事件、原始检测/track ID、处理耗时、回退时段。并保存最终竖屏视频。

BoT-SORT 在这里优先于裸 ByteTrack 的理由是转播机位会摇摄/变焦，前者提供相机运动补偿和可选 ReID。第一轮关闭额外 ReID，避免未验证的人体外观编码器成为额外变量；如果同队球员交换 ID 是主因，再增加 ReID 变体。ByteTrack 本身无相机运动补偿与外观模型，适合作为最低开销对照。当前 Ultralytics 默认 tracker 已是 TrackTrack，必须显式写 `botsort.yaml` / `bytetrack.yaml` 并锁包版本。[官方实现文档](https://docs.ultralytics.com/modes/track)

建议只增加一个较强视觉变体：**SAM 3.1 概念检测/跟踪 + 同一规则导演 + 同一裁切控制器**。使用预先固定的球/球员概念提示，不给测试帧人工纠错；记录实际查询、检测与传播次数。如另做“人工首帧框 + SAMURAI”实验，应明确命名为辅助初始化的跟踪诊断，不与全自动系统混在同一排行榜。

## 因果性、风险与成本核算

| 项目 | 设计约束 / 待测问题 |
|---|---|
| 零前瞻直播 | 模型、记忆、ASR、候选图、控制器只能访问已到达内容；运行时间还必须计入实际输出延迟。模型架构称 streaming 不代表其视频文件 loader 无需等待完整文件 |
| SAM 2 因果性 | 官方 API 有 forward/reverse 传播；直播必须仅前向，并禁止未来帧提示或后验修正已发出的画面。限制条件与 adapter 行为要记录 |
| 镜头切换 | 镜头内视觉身份记忆 reset；事件层记忆可保留“刚进球”等语义，但不能把上一机位坐标沿用到新机位 |
| 小球 | 缩放、运动模糊、白色场线、遮挡可能造成漏检或伪检；mask 精细不等于几像素球可辨认 |
| 身份与事件 | 同队球衣相似；正确跟住一人仍可能错过传出的球。要区分 ID switch、关注区域选择错误和构图控制错误 |
| 不可裁切场景 | 球、接球人、球门可能无法同时装入全高 9:16。为所有方法统一是否允许全景缩放填底/留边；纯裁切赛道应记录“关键内容无法同时容纳” |
| 本地资源 | SAM 3 官方前置条件为 Python ≥3.12、PyTorch ≥2.7、CUDA ≥12.6，检查点需申请访问；不能预设当前 Mac 可直接按官方 GPU 路径运行 |
| SAM 2 系部署 | 原仓库当前要求 Python ≥3.10、Torch ≥2.5.1；SAMURAI 与 SAM2Long 的 fork 环境应独立。训练免除不等于推理免成本 |
| 成本 | 按每输入视频分钟统计 decode、resize、检测、关联/分割、规则/模型决策、恢复、平滑、encode；云 GPU 报实际占用时长及单价，本地报硬件/耗时，不虚构美元成本 |
| 速度与内存 | 分开报冷启动/编译/权重加载与稳态，报告端到端 p50/p95、峰值显存和处理实时系数；不能用论文 GPU FPS 替换本机测量 |

因果接口依据：[SAM 2 视频 predictor](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py)。部署依据：[SAM 3 安装要求](https://github.com/facebookresearch/sam3#installation)、[SAM 2 安装要求](https://github.com/facebookresearch/sam2#installation)。其余风险项为足球场景的工程推断，需实测确认。

现有 30 帧适合验证可见球、人、远景/特写/回放观察能力，无法验证抖动、迟滞、恢复时间或事件切换。应从原视频选取连续短片段覆盖传球、带球、射门、进球后反应、遮挡及切镜，并由人工核对关键内容与可接受裁切范围。**不能使用 SAM、检测器或 agent 输出作为评测真值。**
