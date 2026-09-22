from urllib.parse import unquote

import pytest
from fastapi import Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


@pytest.mark.parametrize(
    "model",
    [
        "guanghe/seedance2.0_企业折扣",
        "guanghe/seedance2.5_企业折",
        "guanghe/seedance2.0-fast（企业折扣）",
        "guanghe/seedance2.0-mini（企业折扣）",
        "toapis/seedance-2-5",
    ],
)
def test_deployment_model_name_can_be_written_to_http_headers(monkeypatch, model):
    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "_get_deployment_model_name", staticmethod(lambda _: model))
    headers = ProxyBaseLLMRequestProcessing.get_custom_headers(user_api_key_dict=UserAPIKeyAuth(), hidden_params={})
    response = Response(content=b"accepted")
    response.headers.update(headers)
    actual = response.headers["x-litellm-model-name"]
    assert unquote(actual) == model
    if model.isascii():
        assert actual == model
