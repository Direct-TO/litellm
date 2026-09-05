"""Metadata helpers for retry-safe asynchronous media submission handling."""

from typing import Final, Literal, TypeVar

import httpx

SubmissionOutcome = Literal["rejected", "accepted", "unknown"]

_SUBMISSION_OUTCOME_ATTR: Final = "litellm_submission_outcome"
_PROVIDER_TASK_ID_ATTR: Final = "litellm_provider_task_id"
_DEFINITIVE_REJECTION_STATUS_CODES: Final = frozenset((400, 401, 403, 404, 422, 429))
_DEFINITIVE_503_REJECTION_MARKERS: Final = ("no available channel",)
_ExceptionT = TypeVar("_ExceptionT", bound=Exception)


def mark_submission_outcome(
    exception: _ExceptionT,
    outcome: SubmissionOutcome,
    provider_task_id: str | None = None,
) -> _ExceptionT:
    """Attach provider-submission provenance without changing the public error shape."""
    setattr(exception, _SUBMISSION_OUTCOME_ATTR, outcome)
    if provider_task_id is not None:
        setattr(exception, _PROVIDER_TASK_ID_ATTR, provider_task_id)
    return exception


def copy_submission_metadata(source: object, target: _ExceptionT) -> _ExceptionT:
    """Preserve submission provenance while mapping provider errors to LiteLLM errors."""
    outcome: Final = getattr(source, _SUBMISSION_OUTCOME_ATTR, None)
    if outcome in ("rejected", "accepted", "unknown"):
        setattr(target, _SUBMISSION_OUTCOME_ATTR, outcome)
    provider_task_id: Final = getattr(source, _PROVIDER_TASK_ID_ATTR, None)
    if isinstance(provider_task_id, str) and provider_task_id:
        setattr(target, _PROVIDER_TASK_ID_ATTR, provider_task_id)
    return target


def get_submission_outcome(exception: Exception) -> SubmissionOutcome | None:
    """Return trusted submission provenance attached by a provider adapter."""
    outcome: Final = getattr(exception, _SUBMISSION_OUTCOME_ATTR, None)
    if outcome in ("rejected", "accepted", "unknown"):
        return outcome
    return None


def get_provider_task_id(exception: Exception) -> str | None:
    """Return the provider task id observed before a later failure, when available."""
    provider_task_id: Final = getattr(exception, _PROVIDER_TASK_ID_ATTR, None)
    return provider_task_id if isinstance(provider_task_id, str) and provider_task_id else None


def classify_http_submission_outcome(response: httpx.Response) -> SubmissionOutcome:
    """Classify only explicit pre-acceptance HTTP rejections as safe to resubmit."""
    if response.status_code in _DEFINITIVE_REJECTION_STATUS_CODES:
        return "rejected"
    if response.status_code == 503:
        normalized_body: Final = response.text.casefold()
        if any(marker in normalized_body for marker in _DEFINITIVE_503_REJECTION_MARKERS):
            return "rejected"
    return "unknown"
