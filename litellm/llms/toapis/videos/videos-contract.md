# 网关视频输入与 ToAPIs 映射

网关接受 `resolution`（480p、720p、1080p、2K、4K）、`aspect_ratio`（如 16:9）及有序 `references`（type、url、role）。智能参数省略，文件参考使用 HTTP(S) URL。该契约不与旧像素 `size`、本地 `input_reference` 或原生参考字段混用；非冲突的管理员 `extra_body` 扩展继续保留。

统一 JSON 创建入口仍为 `POST /v1/videos`。新增可选顶层 `operation`：`generate`、`edit`、`extend`；不传或为 null 按 `generate` 处理。操作不能放进 `extra_body`，也不能用原生 `video_operation`、`action`、`duration` 覆盖统一契约。Router 预检与提交复用相同能力校验；不支持的操作在提交前拒绝，即使 `drop_params=true` 也不能降级为生成。

Seedance 2.5 将操作转换为 `video_operation`。编辑和延长都要求至少一条视频参考，使用既有 `references` URL 与顺序；不能混入首尾帧。两者均须省略 `aspect_ratio`，内部发送 `adaptive`。编辑只接受省略 `seconds` 或 `-1`，发送 `duration=-1`；延长接受 `4–30` 或 `-1`，省略时发送 `-1`。延长的 `seconds` 表示输出总时长，不是新增时长。固定比例或不符合操作的时长会明确报错。

HappyHorse 和 Gemini Omni Official 已有的统一视频编辑映射也要求显式 `operation=edit`，不再因传入视频自动切到编辑；省略操作或 `generate` 携带其不支持的生成视频参考时返回错误。其他 ToAPIs 模型尚未适配的显式 `edit/extend` 会被拒绝。原生旧请求形态不作为统一操作接口。

每个模型只接受自身有文档依据的子集，Router 预检与实际提交使用同一映射。不支持时返回参数错误，不删除参数、降档或丢弃素材。参考视频/音频的真实时长、编码、字节大小由供应商继续验证。能力表表示已实现的适配，不代表供应商能力全集；增加模型时必须同时实现字段转换、限制校验和最终 HTTP 请求体测试。

| 模型 | 可用固定档位 | 参考映射 |
| --- | --- | --- |
| gemini-omni-flash | 720p、1080p（1080p 仅横屏） | 最多 3 张普通图 → image_urls；无视频/音频/显式首尾帧 |
| gemini-omni-flash-preview-official | 720p | 最多 10 张普通图 → image_urls；operation=edit 时最多 3 条视频 → video_list；图片与视频互斥；编辑须省略比例 |
| grok-video-1.0 | 480p、720p | 普通图 → reference_images；首图 → image；合计最多 8 张；无尾帧/视频/音频 |
| grok-video-1.5 | 480p、720p | 必须且仅有一张图片 → image（接受 reference 或 first_frame）；无尾帧/视频/音频 |
| seedance-2 | 480p、720p、1080p、4K | image/video/audio_with_roles；首尾帧与普通参考互斥 |
| seedance-2-fast / mini | 480p、720p | 同上，mini 按文档限制视频/音频数量 |
| seedance-2-5 | 480p、720p、1080p | 三类 with_roles；operation 支持 generate/edit/extend；允许纯音频参考生成；首尾帧须智能比例和智能时长 |
| wan3.0-video | 480p、720p、1080p | 普通图 reference_images、视频 video_list、音频 audio_with_roles、首尾 image_with_roles |
| MiniMax-H3 | 2K | 三类 with_roles；首尾帧须智能比例；音频不可单独输入 |
| happyhorse-1.1 | 720p→720P、1080p→1080P | 最多 9 张普通图 → reference_images；首帧 → image_urls，比例由首帧决定；operation=edit 时 1 条视频 → action=video-edit + url，可附最多 5 张普通图 |
| kling-v3 | 720p→std、1080p→pro | 普通图 → reference_images；首尾帧或混合素材 → image_with_roles，保留显式角色和顺序 |
| kling-v3-omni / kling-video-o1 | 720p→std、1080p→pro | metadata.image_list；统一 reference 未携带视频 base/feature 意图，故不猜测视频角色 |
| veo3.1-fast / quality / lite | 720p、1080p、4K | metadata.resolution；两图首尾或三图参考；quality 不接受参考模式 |
| Veo3.1-fast-official / quality-official | 720p、1080p、4K | 首帧 image_urls、尾帧 metadata.lastFrame、普通图 metadata.referenceImages |
| zexapi/omni_flash-10s | 固定 720p、10 秒 | 16:9 / 9:16 → size=1280x720 / 720x1280；最多 7 张普通图或 7 条视频 → images；不自动切换到首尾帧型号 |
| zexapi/omni_flash-10s-fl | 固定 720p、10 秒 | 显式首帧和可选尾帧 → images，按首/尾顺序排列；不接受普通参考图 |

`seconds` 转为 ToAPIs 的 `duration`；新增校验：Gemini 普通版 4/6/10 秒、Official 1–10 秒、Grok 1–15 秒、Seedance 2/fast 4–15 秒或 -1、mini 4–15 秒、2.5 4–30 秒或 -1、Wan 2–30 秒、HappyHorse/Kling v3 3–15 秒。ZexAPI Omni 的 720p 和 10 秒由型号保证，校验后不发送 resolution/duration；省略 seconds 或 -1 表示采用型号固定时长。

Grok 支持 16:9、9:16、1:1、3:2、2:3；Gemini 和 ZexAPI Omni 支持 16:9、9:16。Gemini Official 视频编辑无法兑现指定宽高比，因此显式指定比例时返回清晰错误，不静默忽略。Gemini 两个型号的文档没有显式首尾帧字段，保留对此类角色的拒绝。

未确认契约的模型、供应商或档位明确拒绝新的扩展字段；旧原生请求形态仍沿用既有适配。

## Seedance 2.5 虚拟人像审核恢复

`toapis/seedance-2-5` 在创建视频时收到明确的 HTTP 400、结构化 `PrivacyInformation` 错误，且错误包含 `input image 'content[n]' ... may contain real person` 时，自动调用同一部署地址和凭据的 `private-avatar` 接口。此流程用于 AI 生成的虚拟人物，不能替代真实人物的 H5 认证。

按 Seedance 的文本在 `content[0]`、图片随后排列的结构，将 `content[n]` 映射到 `image_with_roles` / `image_urls` 的第 n 张图片。越界、缺少索引、自定义 `content`、冲突图片字段或被拦截图片已经是 `asset://` 时保留原错误，不猜测、不整批送审。统一 `references` 与本地图片上传均先沿用既有转换，再处理最终供应商请求体。

只送审被标记的 HTTP(S) 图片，同一次恢复中相同 URL 仅上传一次；每个不同 URL 独立建素材组，避免假定它们属于同一角色。所有目标图片达到 `active` 后，在最终供应商请求体内部替换为 `asset://<asset_id>`，保留顺序、角色、提示词和其他参数，再提交视频一次。统一入口仍要求 HTTP(S) 参考 URL，不向其他供应商放开 `asset://`。

审核流程使用 120 秒预算，每次 HTTP 请求的超时按剩余预算和调用方设置收紧，每 5 秒查询一次；失败、未知状态、无效响应和超时均停止生成并返回错误。日志记录图片序号、group_id 和 asset_id。同一调用中审核开始后的错误带可信重执行阻断标记，Router 不再重试、切换部署或 fallback，但仍保留真实的 rejected / accepted / unknown 提交结果。审核请求移除原生成的幂等键，修正后的生成请求使用派生幂等键。

已有任务 ID、已接受任务的 2xx 响应、普通错误、网络超时和状态查询不启动审核恢复。未知视频提交结果不自动再次 POST。此逻辑只处理创建阶段的确定拒绝；已创建任务在后续查询中才失败时，需要上层保存原请求后另行恢复。

接口依据：[ToAPIs 虚拟人像素材](https://docs.toapis.com/docs/cn/api-reference/videos/seedance-2/private-avatar)。2026-09-13 已手动验证原失败请求的两张虚拟人物图片可达到 `active`；代码回归使用模拟 HTTP 响应，不代表已完成 Seedance 2.5 的真实视频生成验收。

2026-09-11 线上 `/model/info` 按 `model_info.blocked=false` 和视频端点核对：启用 13 个视频型号，包括上表 11 个 ToAPIs 型号（Gemini×2、Grok×2、Seedance×4、Wan、HappyHorse、Kling v3）及 2 个 ZexAPI Omni。其余已配置的视频型号处于停用状态，本次不扩展其适配、不修改启用状态。

依据：

- [Gemini Omni](https://docs.toapis.com/docs/cn/api-reference/videos/gemini-omni-flash/generation)
- [Gemini Omni Official](https://docs.toapis.com/docs/cn/api-reference/videos/gemini-omni-flash-preview-official/generation)
- [Grok 1.0](https://docs.toapis.com/docs/cn/api-reference/videos/grok-video/generation)
- [Grok 1.5](https://docs.toapis.com/docs/cn/api-reference/videos/grok-video-1.5/generation)
- [ZexAPI Omni](https://6l0ket291i.apifox.cn/462450037e0)
- [Seedance 2](https://docs.toapis.com/docs/cn/api-reference/videos/seedance-2/generation)
- [Seedance 2.5](https://docs.toapis.com/docs/cn/api-reference/videos/seedance-2-5/generation)
- [Wan 3.0](https://docs.toapis.com/docs/cn/api-reference/videos/wan3.0/generation)
- [MiniMax H3](https://docs.toapis.com/docs/cn/api-reference/videos/minimax-h3/generation)
- [HappyHorse](https://docs.toapis.com/docs/en/api-reference/videos/happyhorse/generation)
- [Kling v3](https://docs.toapis.com/docs/cn/api-reference/videos/kling-v3/generation)
- [Kling v3 Omni](https://docs.toapis.com/docs/cn/api-reference/videos/kling-v3-omni/generation)
- [Kling O1](https://docs.toapis.com/docs/cn/api-reference/videos/kling-video-o1/generation)
- [Veo](https://docs.toapis.com/docs/cn/api-reference/videos/veo3/generation)
- [Veo Official](https://docs.toapis.com/docs/en/api-reference/videos/veo3-official/generation)
