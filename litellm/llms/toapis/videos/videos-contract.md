# 网关视频输入与 ToAPIs 映射

网关接受 `resolution`（480p、720p、1080p、2K、4K）、`aspect_ratio`（如 16:9）及有序 `references`（type、url、role）。智能参数省略，文件参考使用 HTTP(S) URL。该契约不与旧像素 `size`、本地 `input_reference` 或原生参考字段混用；非冲突的管理员 `extra_body` 扩展继续保留。

每个模型只接受自身有文档依据的子集，Router 预检与实际提交使用同一映射。不支持时返回参数错误，不删除参数、降档或丢弃素材。参考视频/音频的真实时长、编码、字节大小由供应商继续验证。

| 模型 | 可用固定档位 | 参考映射 |
| --- | --- | --- |
| seedance-2 | 480p、720p、1080p、4K | image/video/audio_with_roles；首尾帧与普通参考互斥 |
| seedance-2-fast / mini | 480p、720p | 同上，mini 按文档限制视频/音频数量 |
| seedance-2-5 | 480p、720p、1080p | 三类 with_roles；允许纯音频参考；首尾帧须智能比例和智能时长 |
| wan3.0-video | 480p、720p、1080p | 普通图 reference_images、视频 video_list、音频 audio_with_roles、首尾 image_with_roles |
| MiniMax-H3 | 2K | 三类 with_roles；首尾帧须智能比例；音频不可单独输入 |
| happyhorse-1.1 | 720p、1080p | 普通参考图与单首帧；源视频编辑需单独 operation，不猜测为生成 |
| kling-v3 | 720p→std、1080p→pro | 显式首/尾 image_urls |
| kling-v3-omni / kling-video-o1 | 720p→std、1080p→pro | metadata.image_list；统一 reference 未携带视频 base/feature 意图，故不猜测视频角色 |
| veo3.1-fast / quality / lite | 720p、1080p、4K | metadata.resolution；两图首尾或三图参考；quality 不接受参考模式 |
| Veo3.1-fast-official / quality-official | 720p、1080p、4K | 首帧 image_urls、尾帧 metadata.lastFrame、普通图 metadata.referenceImages |

未确认契约的模型、供应商或档位明确拒绝新的扩展字段；旧原生请求形态仍沿用既有适配。特别是 ZexAPI 当前未取得可验证的新分辨率/多模态角色契约，不能宣称支持。

依据：

- [Seedance 2](https://docs.toapis.com/docs/cn/api-reference/videos/seedance-2/generation)
- [Seedance 2.5](https://docs.toapis.com/docs/cn/api-reference/videos/seedance-2-5/generation)
- [Wan 3.0](https://docs.toapis.com/docs/cn/api-reference/videos/wan3.0/generation)
- [MiniMax H3](https://docs.toapis.com/docs/cn/api-reference/videos/minimax-h3/generation)
- [HappyHorse](https://docs.toapis.com/docs/en/api-reference/videos/happyhorse/generation)
- [Kling v3](https://docs.toapis.com/docs/en/api-reference/videos/kling-v3/generation)
- [Kling v3 Omni](https://docs.toapis.com/docs/cn/api-reference/videos/kling-v3-omni/generation)
- [Kling O1](https://docs.toapis.com/docs/cn/api-reference/videos/kling-video-o1/generation)
- [Veo](https://docs.toapis.com/docs/cn/api-reference/videos/veo3/generation)
- [Veo Official](https://docs.toapis.com/docs/en/api-reference/videos/veo3-official/generation)
