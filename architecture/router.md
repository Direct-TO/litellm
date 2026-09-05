# Router 部署选择

从 [请求路径与路由分流](request-routing.md) 中的 Router 节点进入本图。输入包括 `model`、请求操作类型及参数；本图展开当前 Proxy 使用的异步 Router 选路流程。

```mermaid
flowchart TB
    INPUT["路由输入<br/>model + 操作类型 + 请求参数"]
    RESOLVE["解析模型名 / 别名 / 已注册部署 ID<br/>从当前配置取得候选部署"]
    SCOPE["应用适用的范围限制<br/>访问范围 / Team / Web search"]
    KIND{"请求操作类型？"}
    IMAGE_GEN["图片生成过滤<br/>是否支持 images/generations<br/>是否兼容本次生成参数"]
    IMAGE_EDIT["图片编辑过滤<br/>是否支持 images/edits<br/>是否兼容本次编辑参数"]
    FORM{"候选是具体部署<br/>还是部署列表？"}
    FIXED["具体部署<br/>检查是否被暂停 blocked"]
    AVAILABLE["部署列表：可用性筛选<br/>健康检查：按配置<br/>cooldown → blocked"]
    CONDITIONS["继续筛选候选：按配置<br/>回调 / pre-call / tags / plugins<br/>order / 已尝试部署排除"]
    STRATEGY["按配置的路由策略<br/>选择一个可用部署"]
    CALL["准备调用<br/>注入实际模型 / api_base / api_key<br/>执行适用的并发与限流检查"]
    PROVIDER["执行当前请求对应的 Provider 操作<br/>生成 / 查询状态 / 下载内容等"]
    RESULT["返回本次调用结果"]
    ROUTE_ERROR["路由错误<br/>无兼容图片接口或参数：400<br/>模型不存在 / 暂停 / 无可用部署：对应错误"]
    RECOVERY{"错误类型与配置<br/>是否允许继续恢复？"}
    RETRY["按适用规则重试 / 故障转移<br/>重试当前模型组<br/>或切换配置的候选范围 / fallback 模型"]
    ERROR["返回最终错误"]

    INPUT --> RESOLVE
    RESOLVE --> SCOPE
    RESOLVE -->|"找不到候选模型 / 部署"| ROUTE_ERROR
    SCOPE --> KIND
    KIND -->|"图片生成"| IMAGE_GEN
    KIND -->|"图片编辑"| IMAGE_EDIT
    KIND -->|"其他调用"| FORM
    IMAGE_GEN -->|"保留兼容候选"| FORM
    IMAGE_EDIT -->|"保留兼容候选"| FORM
    IMAGE_GEN -->|"全部不兼容"| ROUTE_ERROR
    IMAGE_EDIT -->|"全部不兼容"| ROUTE_ERROR
    FORM -->|"model 命中已注册部署 ID"| FIXED
    FORM -->|"模型组等列表候选"| AVAILABLE
    FIXED -->|"未暂停：直接使用"| CALL
    FIXED -->|"已暂停"| ROUTE_ERROR
    AVAILABLE --> CONDITIONS
    CONDITIONS -->|"仍有候选"| STRATEGY
    CONDITIONS -->|"无可用候选"| ROUTE_ERROR
    STRATEGY --> CALL
    CALL --> PROVIDER
    CALL -->|"调用前检查失败"| RECOVERY
    PROVIDER -->|"成功"| RESULT
    PROVIDER -->|"失败"| RECOVERY
    ROUTE_ERROR --> RECOVERY
    RECOVERY -->|"允许且未耗尽"| RETRY
    RETRY -.->|"重新执行选路"| RESOLVE
    RECOVERY -->|"不允许 / 无配置 / 已耗尽"| ERROR

    classDef input fill:#e8f1ff,stroke:#4b74b8,color:#172b4d
    classDef decision fill:#fff4d6,stroke:#b58a24,color:#493b18
    classDef image fill:#e9f7ed,stroke:#4b8c60,color:#21452b
    classDef output fill:#eee9fb,stroke:#8270b0,color:#372a57
    classDef error fill:#fdecec,stroke:#bd6666,color:#6b2929
    class INPUT input
    class KIND,FORM,RECOVERY decision
    class IMAGE_GEN,IMAGE_EDIT image
    class PROVIDER,RESULT output
    class ROUTE_ERROR,ERROR error
```

- **图片专属过滤**：当前接口能力映射覆盖图片生成和图片编辑；其他操作跳过这两类过滤。未声明接口能力的旧配置兼容放行；没有待检图片参数时，相应参数检查放行。
- **固定部署与模型组**：`model` 命中已注册部署 ID 时，仍执行适用的图片检查和 `blocked` 检查，随后跳过列表的健康、cooldown、策略选择。`specific_deployment=True` 按实际模型名寻找部署列表，不等同于固定唯一部署 ID。
- **可用性与恢复**：图中展示筛选主顺序；健康路由在部分条件下会恢复候选，不能把它理解为每轮都严格剔除所有不健康部署。重试和 fallback 受错误类型、配置与次数约束；重新选路不保证更换部署。
- **配置来源**：候选部署来自 Router 当前加载的 YAML / DB 模型配置。供应商尺寸规格、计费、日志及结果存储不在本图展开。

源码定位：[候选获取与部署 ID 分支](../litellm/router.py)（`_common_checks_available_deployment`）、[异步筛选与策略选择](../litellm/router.py)（`async_get_available_deployment`）、[Provider 调用](../litellm/router.py)（`_ageneric_api_call_with_fallbacks`）。
