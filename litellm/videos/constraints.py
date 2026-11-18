"""Effective catalogue and preflight constraints for canonical video creation."""

from collections.abc import Mapping
from typing import cast

from litellm.types.videos.constraints import VideoGenerationLimits, VideoVariant


def deployment_video_variants(physical_model: str, limits: object = None) -> list[VideoVariant] | None:
    provider, _, model = physical_model.partition("/")
    if provider == "toapis":
        from litellm.llms.toapis.videos.gateway_contract import generation_variants
    elif provider == "guanghe":
        from litellm.llms.guanghe.videos.transformation import generation_variants
    else:
        return None
    variants = generation_variants(model)
    if limits is None or variants is None:
        return variants
    policy = VideoGenerationLimits.model_validate(limits)
    effective: list[VideoVariant] = []
    for base in variants:
        variant = base.copy()
        # An omitted field delegates the choice to the supplier. Require an explicit
        # value when restricting that dimension, including smart (-1) durations.
        if policy.aspect_ratios is not None:
            variant["aspect_ratio"] = [v for v in base["aspect_ratio"] if v in policy.aspect_ratios]
        if policy.resolutions is not None:
            variant["resolution"] = [v for v in base["resolution"] if v in policy.resolutions]
        if policy.min_seconds or policy.max_seconds:
            variant["seconds"] = [
                s
                for s in base["seconds"]
                if s is not None
                and s > 0
                and (not policy.min_seconds or s >= policy.min_seconds)
                and (not policy.max_seconds or s <= policy.max_seconds)
            ]
        if variant["aspect_ratio"] and variant["resolution"] and variant["seconds"]:
            effective.append(variant)
    return effective


def enforce_video_limits(variants: list[VideoVariant], params: Mapping[str, object]) -> None:
    references = params.get("references")
    frames = isinstance(references, list) and any(
        isinstance(ref, Mapping) and cast(Mapping[str, object], ref).get("role") in ("first_frame", "last_frame")
        for ref in cast(list[object], references)
    )
    seconds = params.get("seconds")
    seconds = int(str(seconds)) if seconds is not None else None
    if not any(
        variant.get("frames", frames) == frames
        and params.get("aspect_ratio") in variant["aspect_ratio"]
        and params.get("resolution") in variant["resolution"]
        and seconds in variant["seconds"]
        for variant in variants
    ):
        raise ValueError("Video duration, aspect ratio or resolution is outside this deployment's generation limits")
