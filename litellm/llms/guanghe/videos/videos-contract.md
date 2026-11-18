# 光盒视频通道

接口依据：光盒 API 文档 https://gc.aitoken.name/#/api-docs，以及 2026-09-16 的 `/models`；Fast 国内key100的规格另于 2026-11-17 使用线上同一渠道凭据核实。

默认 API base 为 `https://newapi.aitoken.name/api/inspiration-waterfall/v1`。使用 `GUANGHE_API_KEY`，可通过 `GUANGHE_API_BASE` 或部署的 `api_base` 覆盖地址。凭据只从运行环境读取。

| LiteLLM 配置名称 | 上游 model_id | 分辨率 | 时长 |
| --- | --- | --- | --- |
| seedance-2 | seedance2.0_企业折扣 | 480p/720p/1080p | 4–15 秒或 -1 |
| seedance-2-5 | seedance2.5_企业折 | 480p/720p/1080p | 4–30 秒或 -1 |
| seedance-2-fast | seedance2.0-fast（国内key100） | 480p/720p | 4–15 秒或 -1 |
| seedance-2-mini | seedance2.0-mini（企业折扣） | 480p/720p | 4–15 秒或 -1 |

四个对外名称分别是独立模型组，每组包含光盒 `order: 1` 和 ToAPIs `order: 2`。先排除暂停、冷却和不支持本次参数的部署，再按 order 选择；光盒可用且参数兼容时优先。旧的 `guanghe-seedance-*` 单独名称不再出现在默认配置中，供应商前缀只保留在内部部署模型。已生成任务的状态查询和下载根据任务 ID 中的供应商与模型筛选部署，不会重新按优先级切到另一家；找不到原供应商时明确报错。

Fast 的旧 `seedance2.0-fast（企业折扣）` 当前已不在供应商可用目录。迁移时保留旧部署的 ID、模型名、地址和凭据，将 `model_info.supported_endpoints` 设为空列表、`order` 设为99，并开启 `disable_background_health_check`，用于已有任务的状态查询与下载。适配器同时在创建参数校验阶段拒绝旧模型，保证直接SDK调用或同步路由也不会向已退休模型提交；读取不经过这项创建校验。不要将旧部署标为 `blocked`，该标记也会阻止历史读取。新国内key100部署继续使用公开组名 `seedance-2-fast`、`order: 1`，ToAPIs 保持备用。此只读旧部署不计入可创建任务的两个渠道。

沿用现有媒体切换规则：只有明确拒绝且符合 403/429/503 策略时才尝试备用；任务已受理、结果未知或人像内容拒绝不会自动再次生成。此配置修改不等于线上 DB Models 已更新。

## 请求和素材

LiteLLM 接受 `seconds`、`resolution`、`aspect_ratio`、`references`，转换为 `POST /video/tasks` 的 `{model_id, prompt, params}`。支持 16:9、9:16、1:1、4:3、3:4、21:9；比例省略映射为 adaptive。时长默认 4 秒，分辨率默认 720p。

声音开关：国内key100声明原生 `generate_audio` 布尔参数，适配器将 `true` 和 `false` 原样传递，非布尔值在提交前拒绝。省略该参数时不补值。其余旧通道的已验收输出默认包含音轨，适配器将星拓造境的 `generate_audio: true` 对应到此默认行为，不发送其 schema 未声明的声音字段；显式 `false` 仍被这些旧通道预检拒绝，由同组支持静音的 ToAPIs 服务。

普通参考按类型映射为 `imageUrls`、`videoUrls`、`audioUrls`，保留各类型内顺序。国内key100的上限分别是 9/3/3，其他既有模型保持 30/10/10；超限在提交前拒绝。首尾帧映射为 `firstFrameUrl`、`lastFrameUrl`，不能混用普通参考。拒绝像素尺寸、未经声明的模型及原生字段覆盖，不静默丢弃参考或降级分辨率。国内key100在供应商网页中的默认值为5秒/480p/16:9；网关仍保持已有4秒/720p/adaptive默认行为，调用方显式参数优先。

```json
{
  "model": "seedance-2-5",
  "prompt": "让参考图中的人物平稳腾空飞起",
  "seconds": "4",
  "resolution": "480p",
  "aspect_ratio": "3:4",
  "references": [{"type": "image", "role": "reference", "url": "https://example.com/person.jpg"}]
}
```

调用方仍传入可访问的 HTTP(S) 素材 URL。LiteLLM 在光盒创建视频前，自动下载普通参考图片、参考视频以及首尾帧，再使用同一光盒部署地址和凭据调用 `POST /files/upload`（`file_type=input_material`、`source=upload`）。上传成功后优先使用 `signed_url`，否则使用 `url`，替换最终供应商请求里的 `imageUrls`、`videoUrls`、`firstFrameUrl`、`lastFrameUrl`；素材顺序、首尾帧角色、提示词和其他参数保持不变。音频仍按原 URL 传递。

同步和异步 SDK/代理请求使用同一准备流程。同一次请求中同类型、同 URL 素材只下载上传一次；不跨请求、账户缓存临时签名地址。文生视频不触发上传。旧 SDK `input_reference` 图片字节沿用既有上传钩子，仅上传一次；遵守网关通用规则，不能与 `resolution/aspect_ratio/references` 混传。

单文件最大 50 MiB，流式读取时同时检查声明大小和实际大小；空文件、错误页、与引用类型不符的 MIME、下载失败或上传失败均阻止生成，不退回原 URL。下载初始地址及每次重定向都复用用户 URL 安全校验，支持管理员的 `user_url_allowed_hosts`；独立下载客户端不携带光盒密钥、生成幂等键或调用者额外请求头。上传判断 HTTP 状态与业务 `success` 字段，不因 HTTP 200 就视为成功；有 `status` 时必须为 `ready`。上传不携带生成幂等键且不跟随重定向，生成提交保持原幂等键。

下载与上传共享本次准备阶段的超时预算。任何准备失败都标记为“尚未提交视频”，阻止外层自动重试及 fallback；已完成部分上传但后续素材失败时也不创建视频。准备成功后立即提交，以免文档默认 10 分钟有效的签名 URL 过期。生成传输层若重发，使用已经准备好的同一请求正文，不再次上传。

此流程是普通文件上传，不是虚拟人像审核。不会根据提示词自动修改比例或时长；视频编辑需要调用方传 `seconds=-1` 并省略 `aspect_ratio`，让光盒获得 `duration=-1`、`aspectRatio=adaptive`。

这四条通道未声明自动虚拟资产上传。本适配不自动把真人照片提交为虚拟人物，不提供未经文档确认的 H5 认证或绕过审核逻辑。上游人像拒绝按失败返回。

## 异步状态与下载

创建接受 HTTP 202 的 `data.job_id`，也接受带 `data.task_id` 的响应。LiteLLM ID 编码 provider、model 和任务 ID；通过 `/v1/videos/{id}` 查询、`/v1/videos/{id}/content` 下载（上游 `/video/tasks/{id}/download`）。跨域下载重定向不携带 API Authorization。

- accepting/unknown/pending/queued → queued，继续查询同一任务。
- processing/in_progress → in_progress。
- success/succeeded/completed → completed，必须有绝对视频 URL。
- failed/cancelled/manual_review → failed，保留具体错误。manual_review 不表示退款已经完成。

每次创建自动携带 UUID `Idempotency-Key`，也可通过 `extra_headers` 指定。底层连接重发沿用相同键；不在适配器内重新创建任务。超时、5xx、无效响应或已经有任务 ID 后的错误标记为 unknown/accepted，阻止 Router 重试或跨部署故障转移。部署配置 `num_retries: 0`。

`credits_cost` 仅保留为供应商元数据，没有可信货币换算时不注册虚假的零价格。创建成功、审核通过、视频生成成功和真实输出规格需要分别验证。

## 2026-11-17 Fast 国内key100验证

迁移前使用同一渠道凭据直连创建一次5秒、480p、16:9、`generate_audio: true`的文生视频，返回任务ID后轮询至 `success`，供应商记录为16积分。下载成片的视频轨时长5.0417秒、864×496、24fps，包含非静音AAC立体声音轨。

部署并更新线上DB后，通过公开 `seedance-2-fast` 和 `/v1/videos` 再提交一次5秒、480p、16:9、`generate_audio: false`的文生视频。响应任务ID明确编码国内key100，约131.5秒轮询到 `completed`，`/content` 下载HTTP 200；文件为5.0417秒、864×496、24fps，无音频轨。线上同时验证旧企业折扣任务及新直连任务均可查询为 `completed`。相关220项回归通过，覆盖参数透传、素材数量边界、新建与旧读取隔离、备用及配置目录。首尾帧、多模态参考和其他规格仍只有契约及模拟回归验证，不代表均已真实生成验收。

## 2026-09-16 人像视频拒绝复现与虚拟资产接口边界

依据用户提供的 `光盒.mhtml`、当前部署 Key 的实时 `/models` 与 `/virtual-assets` 响应：

- 文档中的 `POST /virtual-assets` 接收图片/视频文件，返回的素材需达到 `active` 才能用于 `params.virtualAssetIds`；此接口并非所有模型均可用。
- 对 `seedance2.5_企业折` 实际调用 `GET /virtual-assets?model_id=...&page_size=1` 返回 HTTP 400、`code=invalid_params`、`message=Virtual assets are not supported for this model`。该模型的 schema 没有 `virtualAssetIds`，`auto_virtual_assets=false` 不能解释为支持手动上传。
- `sd_2.5_special_v1` 声明自动虚拟资产上传，但 `maxVideo=0`、资产类型仅 Image/Audio，不能承接本次视频参考。`dreamina-seedance-2-5-hc-ad` 声明自动虚拟资产上传且支持 Video，但属于不同模型渠道；尚未进行该渠道的真实生成验收，不应静默切换。
- 使用已部署 LiteLLM SDK，对原请求 `0ab59372-b097-45e7-8230-890b853cebf3` 进行一次授权重放，保留原提示词、原参考视频、10 秒、480p、16:9、声音开启。重放前以 Range GET 确认原签名视频 URL 返回 HTTP 206。
- 重放幂等键为 `guanghe-repro-5af712cb03134bbba3a90dde638572fe`。创建返回 `queued`，上游任务 ID 为 `task_tyi3i02W04zHZwSQFnpcsSZVdANicH9C`；随后查询返回 HTTP 422、`code=upstream_rejected`，内层错误为 `InputVideoSensitiveContentDetected.PrivacyInformation`，指出输入视频 `content[0]` 可能包含真人。
- 因此，人像拒绝也会在已返回任务 ID 后通过查询接口暴露。只处理创建阶段 HTTP 400/422 的恢复不能覆盖该路径；已有任务应继续保留身份，查询失败不能被当作一次新的、可任意重建的提交。

本次未调用虚拟资产创建接口、未新增自动送审实现、未部署变更，也没有重放成功的生成结果。实现企业折渠道送审恢复的前提是供应商为该模型开放相应接口并确认素材替换契约，或明确选择支持虚拟资产的新模型渠道后单独验证。
