# 足球回放竖屏重构：接口核验

核验日期：2026-11-14。仅查阅官方文档，未调用付费接口、未运行视频实验。以下将接口事实、设计建议和待验证能力分开。

## 已确认的接口事实

- Drex 的 `POST /v1/systemone` 支持 `noul`、`choice`、`score`；原生 `box` 问题明确不支持。`choice` 的选项键作为返回标签，因此可以让模型选择程序已知坐标的候选窗口。来源：[Questions](https://console.nace.ai/docs/drex/guides/questions)。
- 当前文档中的视觉模型为 `drex-max-v1.1`。一次最多 8 张图；每个问题看到同一组、按数组顺序排列的图，可以用 `Image 1` 等引用。图片通过 `images` 提交，普通 URL 文本不会被抓取。JSON 请求体最多 4 MiB，内联 base64 计入；图片会缩放到最多 1,638,400 像素。官方描述的输入模态为文本与图片，未在该接口看到原始音频或视频输入。来源：[Drex Max](https://console.nace.ai/docs/drex/guides/drex-max)。
- `choice` 返回选项、各选项概率和 `confidence`；`confidence` 与最大选项概率不是同一个值。应记录完整分布，不能把模型信心直接当作足球裁切准确率。`usage.input_tokens` 包含图片 token，`usage.image_tokens` 是其中子集，计费时不能重复相加。来源：[Reading answers](https://console.nace.ai/docs/drex/guides/reading-answers)。
- Max v1.1 的整个打包请求上限为 262,144 token；上游 evaluate timeout 是 55 秒，这是超时上限，不能当作实际响应时延。来源：[Limits](https://console.nace.ai/docs/drex/reference/limits)。
- 此处 NDI 指 Nace Document Intelligence，通过 Perception 提供文件处理 API；可与 Drex 共用 console API key。来源：[Perception overview](https://console.nace.ai/docs/perception/documents)。
- Perception 支持 MP4/MOV/AVI/MKV 的 Parse 与 Ground；格式矩阵明确写明音视频 Parse 执行转写，音视频不支持直接 Split/Classify/Extract。来源：[Format support](https://console.nace.ai/docs/perception/format-support)。
- 音视频 Parse 返回转写链接；按 job id 读取时可得到 `result.transcript`，含 speakers，以及带 `start_ms` / `end_ms` 的 speech/silence 分段。因此它可为解说提供时间线证据。文档没有据此承诺球员/足球定位、动作识别或镜头切分能力。来源：[Parse — audio and video](https://console.nace.ai/docs/perception/parse#spreadsheets-audio-and-video)。

## 对最小实验的建议（设计推论，未实测）

1. 先固定一段包含进攻至射门结果的短片，验证竖屏构图；长视频高光检索作为上游独立阶段，避免检索错误掩盖裁切能力。
2. NDI Parse 或 Whisper 任选其一提供带时间戳解说；声音强弱由普通信号处理提取。解说只能作为候选事件线索，其时间戳不等于触球或射门的真实时刻。
3. 程序抽帧、检测镜头边界并生成候选窗口；Drex 根据多帧画面与文本选择窗口、保持或回退。8 张图片预算需要同时容纳时序证据与候选图，不能把全部预算只用作当前帧候选。
4. 例如每次使用 3 张带时间戳的上下文帧 + 4 张候选裁切图，共 7 张。离线允许前看，但必须显式标注；直播实验不得使用未来帧。此分配只是待测配置，并不保证足够描述传球或射门。
5. 模型给出语义目标/窗口偏好，程序负责运动约束和逐帧渲染。原视频切镜头时重新规划；同一镜头内限制速度、加速度和反复换向。固定 9:16 不能容纳关键关系时允许保留完整横屏画面置于竖屏画布，而不是强行裁掉接球队员或球门。
6. 记录完整耗时、模型 usage、转写费、抽帧/编码/恢复成本；与固定中心裁切、规则运动方案共用同一平滑器及回退策略。

## 仍然未知

- Drex 对远景小球、传球方向、遮挡、射门结果、连续动作和候选构图的实际可靠性。
- NDI 对足球解说的人名、噪声和语音时间线准确性，以及转写吞吐量。
- 在相同质量约束下，Drex 决策是否优于规则、是否降低总成本或总延时。
- 真实账号的模型可用性、限流与并发条件，需实现前通过该账号的模型列表核对。

下一步应先用人工标注的少量短片验证以上问题，不能把接口支持多图等同于已经支持可靠的视频理解。
