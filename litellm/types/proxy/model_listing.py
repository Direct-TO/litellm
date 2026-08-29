"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

ModelCapability: TypeAlias = Literal["text", "image", "video", "audio"]


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `mode`, `capability`, token limits, and
    `metadata` are attached when known or requested.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    mode: NotRequired[str]
    capability: NotRequired[ReadOnly[ModelCapability]]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    metadata: NotRequired[ModelInfoMetadata]
