from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import ConfigGeneralSettings


class _LiteLLMParams(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    model: str
    api_key: str | None = None


class _ModelDeployment(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    model_name: str
    litellm_params: _LiteLLMParams


class _DevConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    model_list: tuple[_ModelDeployment, ...]
    general_settings: ConfigGeneralSettings


_CONFIG_PATH: Final = Path(__file__).resolve().parents[3] / "litellm" / "proxy" / "dev_config.yaml"


def _dev_config() -> _DevConfig:
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        return _DevConfig.model_validate(yaml.safe_load(config_file))


def test_deepseek_is_default_chat_completion_model() -> None:
    config: Final = _dev_config()
    deepseek_deployments: Final = tuple(
        deployment for deployment in config.model_list if deployment.model_name == "deepseek-chat"
    )

    assert config.general_settings.completion_model == "deepseek-chat"
    assert len(deepseek_deployments) == 1
    assert deepseek_deployments[0].litellm_params.model == "deepseek/deepseek-chat"
    assert deepseek_deployments[0].litellm_params.api_key == "os.environ/DEEPSEEK_API_KEY"
