# Modal 能力核验：足球回放到竖屏 Agent

调研日期：2026-11-14。范围：Modal 官方文档与官方示例；仅设计调研，未部署、未调用付费计算、未验证足球效果。下文“建议”均为本项目设计推论，不是 Modal 自带功能。

## 结论

Modal 可以承载媒体处理、ASR、检测/跟踪恢复与异步任务；它不直接提供“足球高光理解 + 自动导演 + 平滑竖屏”的成品。建议让 Drex 做编辑、导演、复核决策，Modal 执行有明确输入输出的工具。角色拥有各自记忆与退出条件，调用顺序由证据和预算决定，不要求每段素材都完整经过所有模型。

第一版保留本地控制器也成立：官方异步任务例子允许任意 Python 应用提交任务，并不要求控制器也部署到 Modal。[官方异步 OCR 示例](https://modal.com/docs/examples/doc_ocr_jobs)

## 五个能直接借用的官方示例

| 官方示例 | 可借用部分 | 在本项目中的边界 |
|---|---|---|
| [批量 Whisper](https://modal.com/docs/examples/batched_whisper) | `@app.cls`、`@modal.enter()` 加载模型、Volume 缓存权重、`@modal.batched` 合并输入 | 用于解说转录和事件线索。示例输出文本；词级时间戳、原视频偏移、专名校正须自己接好。批处理等待会增加单次响应延迟，不能直接照搬其吞吐设置做直播。 |
| [YOLO 微调与推理](https://modal.com/docs/examples/finetune_yolo) | GPU 推理类、权重路径、逐图预测及流式返回；训练与推理分离 | 借用部署结构，先不微调。示例数据是猫狗，不证明足球、小球、远景人物的准确率；跟踪身份、遮挡恢复与裁切仍需实现。SAM 也可采用相同 GPU worker 结构，但本次没有核验到可直接复用的官方足球 SAM 方案。 |
| [异步 OCR Job Queue](https://modal.com/docs/examples/doc_ocr_jobs) | 提交长任务、缓存模型、独立 GPU 函数、前端轮询结果的模式 | 改成 `extract_evidence`、`recover_shot`、`render_clip` 的任务提交方式。Marker 文档 OCR 模型不是足球记分牌 OCR，不能直接把示例模型当记分牌识别器。 |
| [LangGraph + Sandbox Agent](https://modal.com/docs/examples/agent) | 模型生成动作、执行工具、读取结果、评价后继续或结束的闭环；Sandbox 可复用执行环境 | 借用“观察—调用—评价—继续”的结构。当前媒体操作可用预注册函数，无须让模型生成任意 Python。仅未来确需现场生成分析代码时再引入 Sandbox。 |
| [WebRTC + YOLO](https://modal.com/docs/examples/webrtc_yolo) | HTTP 信令、`.spawn()` 创建 GPU peer、Dict 交换连接信息、流会话生命周期对应 FunctionCall | 适合后续实时预览/直播输入。示例是摄像头检测，不是直播足球导演；其模型推理/网络指标不能移作 Drex + ASR + 跟踪的端到端延时承诺。 |

## 建议的工具执行边界

1. **CPU 媒体工具**：读取已上传视频；按 PTS 抽帧/抽音频、检测镜头切换、画网格、生成候选窗、计算运动与画面抖动、FFmpeg 渲染。工具返回带 `video_hash / shot_id / pts / evidence_id` 的证据，不让模型自行发明源视频时间。
2. **GPU 恢复工具**：按一个 shot 或连续 chunk 调用一次，容器内顺序处理帧并持有 tracker/SAM 记忆，返回轨迹与可见性。跨镜头清空跟踪状态。`@modal.enter()` 适合一次加载模型，但容器内存不能作为跨重启、跨副本的唯一业务记忆。[容器生命周期](https://modal.com/docs/guide/lifecycle-functions)
3. **ASR 工具**：离线可并行处理独立音频块，保存原始时间偏移；若 NDI 已提供足够的转录/事件证据，就不重复跑 Whisper。NDI 的准确接口与成本需单独核验。
4. **Drex 控制器**：运行在本地或 CPU 函数，观察工具返回、选择下一步、检查预算。`choice` 应映射到本轮合法候选与预注册工具，例如“再取邻近时间窗证据 / 请求恢复 / 接受候选 / 放弃”，不能假设这是模型原生任意函数调用接口。

并行适合不同视频、不同 shot、独立音频块和渲染任务；同一个 tracker 的前后帧存在状态依赖，不应直接逐帧 `.map()`。官方 `.map()` 面向独立输入；默认结果按输入顺序返回，但这不意味着执行顺序或共享状态安全。[扩缩容与 map](https://modal.com/docs/guide/scale)

## 记忆、重试和任务恢复

- **媒体/证据记忆**：Volume 或对象存储保存不可变文件；以 `job_id/tool/input_hash/version` 分路径。Volume 支持 commit/reload，但同文件并发修改存在 last-write-wins；v2 能增加不同文件的并发写入，仍不提供同文件事务语义。因此不要让多个 Agent 覆写一个 `memory.json` 或共享 SQLite 文件。[Volumes 一致性与 v2](https://modal.com/docs/guide/volumes)
- **工作记忆**：短期片段状态、待处理任务、近期镜头意图可放 Dict 或控制器内存；正式决策账本应持久化。Dict 值需要显式写回，闲置七天的对象会过期，读写有网络开销，不适合逐帧热点状态。[Dicts](https://modal.com/docs/guide/dicts)
- **任务记忆**：保存输入哈希、工具版本、状态、FunctionCall ID、结果路径、重试次数、费用。`.spawn()` 返回调用句柄，可用 `FunctionCall.from_id(...).get(timeout=0)` 轮询；结果读取期限为完成后七天，不能当长期归档。[Job processing](https://modal.com/docs/guide/job-queue)
- **恢复语义**：函数异常可重试，容器崩溃会重新调度工作。由此建议做幂等工具与显式检查点：已存在且校验通过的输出复用，提交与完成分别记录；外部模型请求重复执行也可能重复收费。不能由“自动重试”推导整个 Agent 事务 exactly-once。[Failures and retries](https://modal.com/docs/guide/retries)

## 冷启动与完整成本

**GPU 等 Drex 并不免费。** 官方冷启动文档明确：容器 idle 期间占用的 GPU reservation、残留内存等资源仍收费；`min_containers`、`buffer_containers`、`scaledown_window` 在延时与成本间取舍。应让 CPU 控制器等待外部模型，GPU 只执行连续计算块。先采用小并发上限、无常驻热池；演示若预热，单独记录预热和保活费用。[Cold start performance](https://modal.com/docs/guide/cold-start)

定价页 FAQ 明确计费包括应用加载、处理输入，以及处理后的容器保活；缩容至零后停止计算计费。不能仅根据首页“idle 免费”的概括估算成本。[Pricing FAQ](https://modal.com/pricing)

建议每次 run 记录：外部 Drex/Gemini/NDI 调用与重试费、GPU 分配秒数、CPU 与内存消耗、冷启动/热池尾部、存储、出网；同时记录墙钟时间、上传时间、处理视频时长、被接受的高光时长。并行任务不能简单相加当端到端延时。Modal 程序化细粒度 billing reports 目前限定 Team/Enterprise；其他方案先使用任务日志与 dashboard 对账。[Billing](https://modal.com/docs/guide/billing)

## 推荐最小实验

先用现有两段 30 分钟视频和人工核对后的短片段，接入 CPU 媒体工具、一个 GPU 恢复工具和可选 Whisper，保留原有平滑器/渲染器。核心比较应是“规则调度 / Gemini 相同接口 / Drex 相同接口”在同一工具库与预算下的结果，避免同时更换云平台、渲染算法、输入范围后无法归因。

特别优先验证：编辑提出高光后，复核角色能否用邻近帧、记分牌变化、射门/过线/庆祝与回放标记，确认源视频时间或主动继续搜证、拒绝错误事件。先处理“找错片段”这个上游失败，再评估更强 GPU 定位是否必要。复核有限次循环、清晰的不足证据状态与预算耗尽退出，由应用实现。

直播另立实验：仅使用已到达帧，配置有界缓冲和过期帧策略，模型决策按低频事件触发、逐帧运动由轻量工具完成。WebRTC 示例可作为传输起点；本轮不引入直播部署，也不把离线未来信息收益当作直播能力。
