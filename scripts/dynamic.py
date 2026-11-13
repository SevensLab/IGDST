"""Predict additions to safety profiles from new accident narratives."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable
from common import (DIMENSIONS, DYNAMIC_DIR, ROOT, call_llm_json, compact_vocab_text,
                    current_profile, jsonl_read, jsonl_write, keyword_matches,
                    last_llm_usage, llm_configured, load_approved_gold_rows, load_dynamic_split,
                    load_env_file, normalize_prediction_tags, now_utc, request_hash, span_exists)

PROMPT_VERSION = "phase3_dynamic_v2"


PROMPT_D2_V1 = (ROOT / "prompts/prompt_d2_v1.txt").read_text(encoding="utf-8")


PROMPT_D3_V1 = (ROOT / "prompts/prompt_d3_v1.txt").read_text(encoding="utf-8")


PROMPT_D4_GENERATOR_V1 = (ROOT / "prompts/prompt_d4_generator_v1.txt").read_text(encoding="utf-8")


PROMPT_D4_VERIFIER_V1 = (ROOT / "prompts/prompt_d4_verifier_v1.txt").read_text(encoding="utf-8")


TEMPERATURE = 0


SCHEMA_MAX_ATTEMPTS = 3


KEYWORD_RULES_PATH = ROOT / "config" / "keyword_rules_v1.json"


OUTPUT_PATHS = {
    "d1": lambda split: DYNAMIC_DIR / f"rule_dynamic_{split}.jsonl",
    "d2": lambda split: DYNAMIC_DIR / f"llm_new_accident_{split}.jsonl",
    "d3": lambda split: DYNAMIC_DIR / f"llm_profile_{split}.jsonl",
    "d4": lambda split: DYNAMIC_DIR / f"igdst_dynamic_{split}.jsonl",
}


METHOD_NAMES = {
    "d1": "d1_rule",
    "d2": "d2_llm_new_accident",
    "d3": "d3_llm_profile",
    "d4": "d4_igdst_dynamic",
}


def usage_estimate(system_prompt: str, user_prompt: str, raw: str = "") -> dict[str, Any]:
    input_tokens = max(1, math.ceil((len(system_prompt) + len(user_prompt)) / 4))
    output_tokens = math.ceil(len(raw) / 4) if raw else 0
    return {
        "api_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_estimated": True,
    }


def usage_from_response(system_prompt: str, user_prompt: str, raw: str = "") -> dict[str, Any]:
    actual = last_llm_usage()
    return actual if actual else usage_estimate(system_prompt, user_prompt, raw)


def add_usage(*parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_calls": sum(int(part.get("api_calls", 0)) for part in parts),
        "input_tokens": sum(int(part.get("input_tokens", 0)) for part in parts),
        "output_tokens": sum(int(part.get("output_tokens", 0)) for part in parts),
        "total_tokens": sum(int(part.get("total_tokens", 0)) for part in parts),
        "usage_estimated": any(bool(part.get("usage_estimated", False)) for part in parts),
    }


class ValidatedCallFailure(RuntimeError):
    """Preserve retry usage when every schema-validation attempt fails."""

    def __init__(
        self,
        context: str,
        errors: list[str],
        usage: dict[str, Any],
        attempts: int,
        raw_response: str = "",
    ) -> None:
        detail = " | ".join(errors)
        super().__init__(
            f"{context} failed after {attempts} schema attempt(s): {detail}"
        )
        self.usage = usage
        self.attempts = attempts
        self.raw_response = raw_response


def call_llm_json_validated(
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[Any], Any],
    context: str,
) -> tuple[Any, str, dict[str, Any], int]:
    """Retry responses that parse as JSON but violate the required schema."""
    aggregate_usage: dict[str, Any] = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_estimated": False,
    }
    errors = []
    last_raw = ""
    for attempt in range(1, SCHEMA_MAX_ATTEMPTS + 1):
        attempt_user_prompt = user_prompt
        if errors:
            attempt_user_prompt += (
                "\n\nRESPONSE SCHEMA CORRECTION:\n"
                "The previous response was invalid: "
                + errors[-1]
                + "\nReturn a fresh JSON object that follows the system-requested "
                "schema exactly. Do not omit or rename required fields."
            )
        try:
            parsed, raw = call_llm_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": attempt_user_prompt},
                ],
                temperature=TEMPERATURE,
                max_retries=3,
            )
            last_raw = raw
        except Exception as exc:
            aggregate_usage = add_usage(
                aggregate_usage,
                usage_estimate(system_prompt, attempt_user_prompt),
            )
            errors.append(f"attempt_{attempt}:api_or_json:{exc}")
            if attempt < SCHEMA_MAX_ATTEMPTS:
                continue
            raise ValidatedCallFailure(
                context, errors, aggregate_usage, attempt, last_raw
            ) from exc

        aggregate_usage = add_usage(
            aggregate_usage,
            usage_from_response(system_prompt, attempt_user_prompt, raw),
        )
        try:
            validated = validator(parsed)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"attempt_{attempt}:{exc}")
            continue
        return validated, raw, aggregate_usage, attempt

    raise ValidatedCallFailure(
        context,
        errors,
        aggregate_usage,
        SCHEMA_MAX_ATTEMPTS,
        last_raw,
    )


def safe_user_prompt(text: str) -> str:
    if "gold_" in text.lower():
        raise RuntimeError("Prediction prompt contains forbidden Gold field text.")
    return text


def profile_block(profile: dict[str, set[str]]) -> str:
    return "\n".join(
        f"Current {dimension}: {', '.join(sorted(profile[dimension])) or '[]'}"
        for dimension in DIMENSIONS
    )


def resource_accident_block(row: dict[str, str]) -> str:
    return (
        f"Resource name: {row['resource_name']}\n"
        f"Resource type: {row['resource_type']}\n"
        f"New accident:\n{row['new_accident_description']}"
    )


def validate_tag_arrays(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    normalized = {}
    for dimension in DIMENSIONS:
        raw = value.get(dimension, [])
        # Some model responses use null or a singleton label for an empty or
        # one-tag dimension. Normalize only these unambiguous schema variants.
        if raw is None:
            raw = []
        elif isinstance(raw, str):
            raw = [raw] if raw.strip() else []
        elif isinstance(raw, dict):
            if "tag" in raw:
                raw = [raw]
            elif isinstance(raw.get("tags"), list):
                raw = raw["tags"]
            else:
                raise ValueError(f"{context}.{dimension} must be an array.")
        elif not isinstance(raw, list):
            raise ValueError(f"{context}.{dimension} must be an array.")
        normalized[dimension] = raw
    return normalized


def compute_add(
    proposed: dict[str, list[str]], profile: dict[str, set[str]]
) -> dict[str, list[str]]:
    return {
        dimension: sorted(set(proposed[dimension]) - profile[dimension])
        for dimension in DIMENSIONS
    }


def action_from_add(additions: dict[str, list[str]]) -> str:
    return "ADD" if any(additions[dimension] for dimension in DIMENSIONS) else "NO_CHANGE"


def replace_row(
    path: Path,
    rows: list[dict[str, Any]],
    case_id: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    updated = [row for row in rows if row.get("dynamic_case_id") != case_id]
    updated.append(record)
    jsonl_write(path, updated)
    return updated


def cached_row(
    rows: list[dict[str, Any]], case_id: str, method: str, expected_hash: str
) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("dynamic_case_id") == case_id
        and row.get("method") == method
        and row.get("request_hash") == expected_hash
        and row.get("status") == "success"
    ]
    return matches[-1] if matches else None


def keyword_incident_tags(row: dict[str, str], rules: dict[str, Any]) -> dict[str, set[str]]:
    text = f"{row['resource_name']} {row['new_accident_description']}".lower()
    predictions = {dimension: set() for dimension in DIMENSIONS}
    for dimension in DIMENSIONS:
        for tag, patterns in rules.get(dimension, {}).items():
            if any(keyword_matches(text, pattern) for pattern in patterns):
                predictions[dimension].add(tag)
    return predictions


def base_record(row: dict[str, str], method: str, model: str) -> dict[str, Any]:
    return {
        "dynamic_case_id": row["dynamic_case_id"],
        "resource_id": row["resource_id"],
        "new_accident_id": row["new_accident_id"],
        "method": method,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "temperature": TEMPERATURE,
        "timestamp": now_utc(),
    }


def run_d1(
    rows: list[dict[str, str]], split: str, vocabulary: dict[str, set[str]]
) -> list[dict[str, Any]]:
    rules = json.loads(KEYWORD_RULES_PATH.read_text(encoding="utf-8"))
    output_path = OUTPUT_PATHS["d1"](split)
    output_rows = jsonl_read(output_path)
    for row in rows:
        profile = current_profile(row)
        incident = keyword_incident_tags(row, rules)
        normalized = normalize_prediction_tags(
            {dimension: sorted(incident[dimension]) for dimension in DIMENSIONS},
            vocabulary,
        )
        additions = compute_add(normalized, profile)
        record = {
            **base_record(row, METHOD_NAMES["d1"], "none"),
            "predicted_action": action_from_add(additions),
            "predicted_add": additions,
            "status": "success",
            "failure_reason": "",
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_estimated": False,
        }
        output_rows = replace_row(
            output_path, output_rows, row["dynamic_case_id"], record
        )
    return output_rows


def d2_user_prompt(row: dict[str, str], vocabulary_text: str) -> str:
    return safe_user_prompt(
        resource_accident_block(row)
        + "\n\nControlled vocabulary:\n"
        + vocabulary_text
    )


def d3_user_prompt(row: dict[str, str], vocabulary_text: str) -> str:
    return safe_user_prompt(
        resource_accident_block(row)
        + "\n\n"
        + profile_block(current_profile(row))
        + "\n\nControlled vocabulary:\n"
        + vocabulary_text
    )


def d4_generator_user_prompt(row: dict[str, str], vocabulary_text: str) -> str:
    """Build the D4 semantic input without exposing Current Profile."""
    return safe_user_prompt(
        resource_accident_block(row)
        + "\n\nControlled vocabulary:\n"
        + vocabulary_text
    )


def run_simple_llm_method(
    method_key: str,
    rows: list[dict[str, str]],
    split: str,
    vocabulary: dict[str, set[str]],
    vocabulary_text: str,
    force: bool,
) -> list[dict[str, Any]]:
    method = METHOD_NAMES[method_key]
    system_prompt = PROMPT_D2_V1 if method_key == "d2" else PROMPT_D3_V1
    output_path = OUTPUT_PATHS[method_key](split)
    output_rows = [] if force else jsonl_read(output_path)
    model = os.environ["LLM_MODEL"]

    for row in rows:
        user_prompt = (
            d2_user_prompt(row, vocabulary_text)
            if method_key == "d2"
            else d3_user_prompt(row, vocabulary_text)
        )
        current_hash = request_hash(method, model, TEMPERATURE, system_prompt, user_prompt)
        cached = None if force else cached_row(
            output_rows, row["dynamic_case_id"], method, current_hash
        )
        if cached is not None:
            continue
        call_usage = usage_estimate(system_prompt, user_prompt)
        try:
            parsed, raw = call_llm_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_retries=3,
            )
            call_usage = usage_from_response(system_prompt, user_prompt, raw)
            if method_key == "d2":
                proposed_obj = validate_tag_arrays(parsed, "D2 response")
                raw_model_action = ""
            else:
                raw_model_action = str(parsed.get("action", "")).strip().upper()
                if raw_model_action not in {"ADD", "NO_CHANGE"}:
                    raise ValueError("D3 action must be ADD or NO_CHANGE.")
                proposed_obj = validate_tag_arrays(parsed.get("add"), "D3 response.add")
            proposed = normalize_prediction_tags(proposed_obj, vocabulary)
            additions = compute_add(proposed, current_profile(row))
            record = {
                **base_record(row, method, model),
                "request_hash": current_hash,
                "raw_model_action": raw_model_action,
                "incident_tags" if method_key == "d2" else "model_add": proposed,
                "predicted_action": action_from_add(additions),
                "predicted_add": additions,
                "raw_response": raw,
                "status": "success",
                "failure_reason": "",
                **call_usage,
            }
        except Exception as exc:
            record = {
                **base_record(row, method, model),
                "request_hash": current_hash,
                "status": "failed",
                "failure_reason": f"api_or_schema_failure:{exc}",
                **call_usage,
            }
        output_rows = replace_row(
            output_path, output_rows, row["dynamic_case_id"], record
        )
    return output_rows


def extract_generator_candidates(
    parsed: Any,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    if not isinstance(parsed, dict):
        raise ValueError("D4 generator response must be an object.")
    raw_action = str(parsed.get("action", "")).strip().upper()
    if raw_action not in {"ADD", "NO_CHANGE"}:
        raise ValueError("D4 generator action must be ADD or NO_CHANGE.")
    add = validate_tag_arrays(parsed.get("add"), "D4 response.add")
    candidates = []
    rejected = []
    for dimension in DIMENSIONS:
        for item in add[dimension]:
            if not isinstance(item, dict):
                rejected.append(
                    {
                        "dimension": dimension,
                        "raw_candidate": str(item),
                        "evidence_normalization": "malformed_candidate_rejected",
                    }
                )
                continue
            tag = str(item.get("tag", "")).strip()
            evidence = str(item.get("evidence_span", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not tag or not evidence or not reason:
                rejected.append(
                    {
                        "dimension": dimension,
                        "tag": tag,
                        "raw_evidence_span": evidence,
                        "reason": reason,
                        "evidence_normalization": "malformed_candidate_rejected",
                    }
                )
                continue
            candidates.append(
                {
                    "dimension": dimension,
                    "tag": tag,
                    "evidence_span": evidence,
                    "reason": reason,
                }
            )
    return raw_action, candidates, rejected


def automatic_filter(
    row: dict[str, str],
    candidates: list[dict[str, str]],
    vocabulary: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    profile = current_profile(row)
    results = []
    eligible_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for candidate in candidates:
        dimension = candidate["dimension"]
        tag = candidate["tag"]
        result: dict[str, Any] = {**candidate}
        if tag not in vocabulary[dimension]:
            result.update(
                {
                    "normalized_tag": f"UNMAPPED::{tag}",
                    "automatic_filter": "UNMAPPED",
                    "verdict": "UNSUPPORTED",
                }
            )
        elif not span_exists(candidate["evidence_span"], row["new_accident_description"]):
            result.update(
                {
                    "normalized_tag": tag,
                    "automatic_filter": "UNSUPPORTED",
                    "verdict": "UNSUPPORTED",
                }
            )
        else:
            result.update(
                {
                    "normalized_tag": tag,
                    "automatic_filter": "PASS",
                    "already_known": tag in profile[dimension],
                    "verdict": "PENDING_VERIFIER",
                }
            )
            eligible_by_key.setdefault((dimension, tag), candidate)
        results.append(result)
    return results, list(eligible_by_key.values())


def split_verbatim_candidates(
    row: dict[str, str],
    candidates: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    accepted = []
    rejected = []
    description = row["new_accident_description"]
    for candidate in candidates:
        span = str(candidate.get("evidence_span", "")).strip()
        if span_exists(span, description):
            accepted.append(candidate)
            continue
        rejected.append(
            {
                **candidate,
                "raw_evidence_span": span,
                "evidence_span": "",
                "evidence_normalization": "nonverbatim_span_rejected",
            }
        )
    return accepted, rejected


def verifier_user_prompt(
    row: dict[str, str],
    eligible: list[dict[str, str]],
    definitions: dict[str, dict[str, str]],
) -> str:
    payload = {
        "resource": row["resource_name"],
        "resource_type": row["resource_type"],
        "new_accident": row["new_accident_description"],
        "candidates": [
            {
                "dimension": candidate["dimension"],
                "tag": candidate["tag"],
                "definition": definitions[candidate["dimension"]].get(
                    candidate["tag"], ""
                ),
                "evidence_span": candidate["evidence_span"],
            }
            for candidate in eligible
        ],
    }
    return safe_user_prompt(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def validate_verifier(
    parsed: Any, eligible: list[dict[str, str]]
) -> list[dict[str, str]]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("verdicts"), list):
        raise ValueError("D4 verifier response.verdicts must be an array.")
    expected = {(item["dimension"], item["tag"]) for item in eligible}
    verdicts = []
    seen = set()
    for item in parsed["verdicts"]:
        if not isinstance(item, dict):
            raise ValueError("D4 verifier verdict must be an object.")
        dimension = str(item.get("dimension", "")).strip()
        tag = str(item.get("tag", "")).strip()
        verdict = str(item.get("verdict", "")).strip().upper()
        key = (dimension, tag)
        if key not in expected or key in seen:
            raise ValueError(f"D4 verifier returned unexpected/duplicate candidate: {key}")
        if verdict not in {"SUPPORTED", "UNSUPPORTED"}:
            raise ValueError(f"D4 verifier returned invalid verdict: {verdict!r}")
        seen.add(key)
        verdicts.append(
            {"dimension": dimension, "tag": tag, "verdict": verdict}
        )
    if seen != expected:
        raise ValueError(f"D4 verifier result set mismatch: {len(seen)}/{len(expected)}")
    return verdicts


def d4_composite_hash(
    model: str, generator_user: str, verifier_user: str
) -> str:
    return request_hash(
        METHOD_NAMES["d4"],
        model,
        TEMPERATURE,
        PROMPT_D4_GENERATOR_V1 + "\n" + PROMPT_D4_VERIFIER_V1,
        generator_user + "\n" + verifier_user,
    )


def reusable_d4_row(
    output_rows: list[dict[str, Any]],
    row: dict[str, str],
    model: str,
    generator_user: str,
    vocabulary: dict[str, set[str]],
    definitions: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    generator_hash = request_hash(
        METHOD_NAMES["d4"], model, TEMPERATURE, PROMPT_D4_GENERATOR_V1, generator_user
    )
    for cached in reversed(output_rows):
        if (
            cached.get("dynamic_case_id") != row["dynamic_case_id"]
            or cached.get("method") != METHOD_NAMES["d4"]
            or cached.get("status") != "success"
            or cached.get("generator_request_hash") != generator_hash
        ):
            continue
        candidates = cached.get("generator_candidates", [])
        if not isinstance(candidates, list):
            continue
        if any(
            not span_exists(
                candidate.get("evidence_span", ""),
                row["new_accident_description"],
            )
            for candidate in candidates
        ):
            continue
        _, eligible = automatic_filter(row, candidates, vocabulary)
        verifier_user = verifier_user_prompt(row, eligible, definitions) if eligible else ""
        expected_verifier_hash = (
            request_hash(
                METHOD_NAMES["d4"],
                model,
                TEMPERATURE,
                PROMPT_D4_VERIFIER_V1,
                verifier_user,
            )
            if eligible
            else ""
        )
        expected_hash = d4_composite_hash(model, generator_user, verifier_user)
        if (
            cached.get("verifier_request_hash") == expected_verifier_hash
            and cached.get("request_hash") == expected_hash
        ):
            return cached
    return None


def reusable_d4_generator_stage(
    output_rows: list[dict[str, Any]],
    row: dict[str, str],
    model: str,
    generator_user: str,
) -> dict[str, Any] | None:
    """Reuse a successful generator stage from a verifier-failed checkpoint."""
    generator_hash = request_hash(
        METHOD_NAMES["d4"],
        model,
        TEMPERATURE,
        PROMPT_D4_GENERATOR_V1,
        generator_user,
    )
    for cached in reversed(output_rows):
        if (
            cached.get("dynamic_case_id") != row["dynamic_case_id"]
            or cached.get("method") != METHOD_NAMES["d4"]
            or cached.get("generator_request_hash") != generator_hash
        ):
            continue
        candidates = cached.get("generator_candidates")
        raw_action = cached.get("raw_model_action")
        if not isinstance(candidates, list) or raw_action not in {
            "ADD",
            "NO_CHANGE",
        }:
            continue
        if any(
            not span_exists(
                candidate.get("evidence_span", ""),
                row["new_accident_description"],
            )
            for candidate in candidates
        ):
            continue
        return cached
    return None


def run_d4(
    rows: list[dict[str, str]],
    split: str,
    vocabulary: dict[str, set[str]],
    vocabulary_text: str,
    definitions: dict[str, dict[str, str]],
    force: bool,
) -> list[dict[str, Any]]:
    output_path = OUTPUT_PATHS["d4"](split)
    output_rows = [] if force else jsonl_read(output_path)
    model = os.environ["LLM_MODEL"]

    for row in rows:
        generator_user = d4_generator_user_prompt(row, vocabulary_text)
        generator_hash = request_hash(
            METHOD_NAMES["d4"],
            model,
            TEMPERATURE,
            PROMPT_D4_GENERATOR_V1,
            generator_user,
        )
        if not force and reusable_d4_row(
            output_rows,
            row,
            model,
            generator_user,
            vocabulary,
            definitions,
        ) is not None:
            continue

        cached_generator = (
            None
            if force
            else reusable_d4_generator_stage(
                output_rows,
                row,
                model,
                generator_user,
            )
        )
        prior_usage: dict[str, Any] = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_estimated": False,
        }

        generator_usage = usage_estimate(PROMPT_D4_GENERATOR_V1, generator_user)
        generator_schema_attempts = 0
        resumed_from_generator_cache = cached_generator is not None
        if cached_generator is not None:
            raw_model_action = cached_generator["raw_model_action"]
            candidates = cached_generator["generator_candidates"]
            rejected_candidates = cached_generator.get(
                "rejected_generator_candidates", []
            )
            generator_raw = cached_generator.get("generator_raw_response", "")
            generator_schema_attempts = int(
                cached_generator.get("generator_schema_attempts") or 1
            )
            generator_usage = {
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "usage_estimated": False,
            }
            prior_usage = {
                "api_calls": int(cached_generator.get("api_calls", 0)),
                "input_tokens": int(cached_generator.get("input_tokens", 0)),
                "output_tokens": int(cached_generator.get("output_tokens", 0)),
                "total_tokens": int(cached_generator.get("total_tokens", 0)),
                "usage_estimated": bool(
                    cached_generator.get("usage_estimated", False)
                ),
            }
        else:
            try:
                def validate_generator(parsed: Any) -> tuple[
                    str, list[dict[str, str]], list[dict[str, str]]
                ]:
                    raw_action, parsed_candidates, malformed = (
                        extract_generator_candidates(parsed)
                    )
                    accepted, rejected = split_verbatim_candidates(
                        row, parsed_candidates
                    )
                    return raw_action, accepted, malformed + rejected

                (
                    generator_result,
                    generator_raw,
                    generator_usage,
                    generator_schema_attempts,
                ) = call_llm_json_validated(
                    PROMPT_D4_GENERATOR_V1,
                    generator_user,
                    validate_generator,
                    "D4 generator",
                )
                raw_model_action, candidates, rejected_candidates = generator_result
            except ValidatedCallFailure as exc:
                record = {
                    **base_record(row, METHOD_NAMES["d4"], model),
                    "request_hash": generator_hash,
                    "generator_request_hash": generator_hash,
                    "verifier_request_hash": "",
                    "status": "failed",
                    "failure_reason": f"generator_failed:{exc}",
                    "generator_schema_attempts": exc.attempts,
                    "generator_raw_response": exc.raw_response,
                    **exc.usage,
                }
                output_rows = replace_row(
                    output_path, output_rows, row["dynamic_case_id"], record
                )
                continue

        filter_results, eligible = automatic_filter(row, candidates, vocabulary)
        verifier_user = verifier_user_prompt(row, eligible, definitions) if eligible else ""
        verifier_hash = (
            request_hash(
                METHOD_NAMES["d4"],
                model,
                TEMPERATURE,
                PROMPT_D4_VERIFIER_V1,
                verifier_user,
            )
            if eligible
            else ""
        )
        final_hash = d4_composite_hash(model, generator_user, verifier_user)
        verifier_results: list[dict[str, str]] = []
        verifier_raw = ""
        verifier_usage: dict[str, Any] = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_estimated": False,
        }
        verifier_schema_attempts = 0
        if eligible:
            verifier_usage = usage_estimate(PROMPT_D4_VERIFIER_V1, verifier_user)
            try:
                (
                    verifier_results,
                    verifier_raw,
                    verifier_usage,
                    verifier_schema_attempts,
                ) = call_llm_json_validated(
                    PROMPT_D4_VERIFIER_V1,
                    verifier_user,
                    lambda parsed: validate_verifier(parsed, eligible),
                    "D4 verifier",
                )
            except ValidatedCallFailure as exc:
                record = {
                    **base_record(row, METHOD_NAMES["d4"], model),
                    "request_hash": final_hash,
                    "generator_request_hash": generator_hash,
                    "verifier_request_hash": verifier_hash,
                    "raw_model_action": raw_model_action,
                    "generator_candidates": candidates,
                    "rejected_generator_candidates": rejected_candidates,
                    "automatic_filter_results": filter_results,
                    "generator_raw_response": generator_raw,
                    "verifier_raw_response": exc.raw_response,
                    "status": "failed",
                    "failure_reason": f"verifier_failed:{exc}",
                    "generator_schema_attempts": generator_schema_attempts,
                    "verifier_schema_attempts": exc.attempts,
                    "resumed_from_generator_cache": resumed_from_generator_cache,
                    **add_usage(prior_usage, generator_usage, exc.usage),
                }
                output_rows = replace_row(
                    output_path, output_rows, row["dynamic_case_id"], record
                )
                continue

        verdict_by_key = {
            (item["dimension"], item["tag"]): item["verdict"]
            for item in verifier_results
        }
        for result in filter_results:
            if result["automatic_filter"] == "PASS":
                result["verdict"] = verdict_by_key[(result["dimension"], result["tag"])]
        profile = current_profile(row)
        additions = {dimension: [] for dimension in DIMENSIONS}
        for verdict in verifier_results:
            if (
                verdict["verdict"] == "SUPPORTED"
                and verdict["tag"] not in profile[verdict["dimension"]]
            ):
                additions[verdict["dimension"]].append(verdict["tag"])
        additions = {
            dimension: sorted(set(additions[dimension])) for dimension in DIMENSIONS
        }
        record = {
            **base_record(row, METHOD_NAMES["d4"], model),
            "request_hash": final_hash,
            "generator_request_hash": generator_hash,
            "verifier_request_hash": verifier_hash,
            "raw_model_action": raw_model_action,
            "generator_candidates": candidates,
            "rejected_generator_candidates": rejected_candidates,
            "automatic_filter_results": filter_results,
            "verifier_results": verifier_results,
            "predicted_add": additions,
            "predicted_action": action_from_add(additions),
            "generator_raw_response": generator_raw,
            "verifier_raw_response": verifier_raw,
            "status": "success",
            "failure_reason": "",
            "generator_schema_attempts": generator_schema_attempts,
            "verifier_schema_attempts": verifier_schema_attempts,
            "resumed_from_generator_cache": resumed_from_generator_cache,
            **add_usage(prior_usage, generator_usage, verifier_usage),
        }
        output_rows = replace_row(
            output_path, output_rows, row["dynamic_case_id"], record
        )
    return output_rows


def parse_methods(value: str) -> list[str]:
    requested = {part.strip().lower() for part in value.split(",") if part.strip()}
    invalid = requested - set(METHOD_NAMES)
    if invalid:
        raise ValueError(f"Unknown methods: {sorted(invalid)}")
    return [method for method in ["d1", "d2", "d3", "d4"] if method in requested]


def parse_case_ids(values: list[str] | None) -> set[str]:
    return {
        item.strip()
        for value in values or []
        for item in value.split(",")
        if item.strip()
    }


def retryable_case_ids(
    split_rows: list[dict[str, str]], methods: list[str], split: str
) -> set[str]:
    """Return only missing, failed, or stale rows for the requested methods."""
    expected_ids = {row["dynamic_case_id"] for row in split_rows}
    retry_ids: set[str] = set()
    for method in methods:
        latest = {
            row.get("dynamic_case_id", ""): row
            for row in jsonl_read(OUTPUT_PATHS[method](split))
            if row.get("dynamic_case_id") in expected_ids
        }
        for case_id in expected_ids:
            record = latest.get(case_id)
            if record is None or record.get("status") != "success":
                retry_ids.add(case_id)
                continue
            if method == "d4" and record.get("prompt_version") != PROMPT_VERSION:
                retry_ids.add(case_id)
    return retry_ids


def select_rows(
    rows: list[dict[str, str]],
    split: str,
    limit: int | None,
    case_ids: set[str],
) -> list[dict[str, str]]:
    if case_ids:
        selected = [row for row in rows if row["dynamic_case_id"] in case_ids]
        missing = case_ids - {row["dynamic_case_id"] for row in selected}
        if missing:
            raise ValueError(f"Unknown dynamic_case_id values: {sorted(missing)}")
        return selected[:limit] if limit is not None else selected
    return rows[:limit] if limit is not None else rows


def vocabulary_assets() -> tuple[dict[str, set[str]], list[dict[str, str]], Path]:
    _, _, vocabulary, vocabulary_rows, workbook_path = load_approved_gold_rows()
    return vocabulary, vocabulary_rows, workbook_path


def vocabulary_definitions(
    vocabulary: dict[str, set[str]], vocabulary_rows: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    definitions = {dimension: {} for dimension in DIMENSIONS}
    for row in vocabulary_rows:
        dimension = row.get("dimension", "")
        tag = row.get("tag", "")
        if dimension in definitions and tag in vocabulary[dimension]:
            definitions[dimension][tag] = row.get("definition", "")
    return definitions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--methods", default="d1,d2,d3,d4")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Run only missing, failed, or stale rows for the requested methods.",
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    split_rows = load_dynamic_split(args.split)
    requested_case_ids = parse_case_ids(args.case_ids)
    if args.retry_failed_only and requested_case_ids:
        parser.error("--retry-failed-only cannot be combined with --case-ids")
    if args.retry_failed_only and args.force:
        parser.error("--retry-failed-only cannot be combined with --force")
    if args.retry_failed_only:
        requested_case_ids = retryable_case_ids(
            split_rows, methods, args.split
        )
        if not requested_case_ids:
            print("No missing, failed, or stale prediction rows to retry.")
            return 0
        print(f"Retrying incomplete case rows={len(requested_case_ids)}")
    selected = select_rows(
        split_rows,
        args.split,
        args.limit,
        requested_case_ids,
    )
    vocabulary, vocabulary_rows, _ = vocabulary_assets()
    vocab_text = compact_vocab_text(vocabulary, vocabulary_rows)
    definitions = vocabulary_definitions(vocabulary, vocabulary_rows)

    load_env_file()
    if any(method in methods for method in ["d2", "d3", "d4"]) and not llm_configured():
        raise RuntimeError("Missing LLM_API_KEY, LLM_BASE_URL, or LLM_MODEL.")

    failed = False
    for method in methods:
        if method == "d1":
            output = run_d1(selected, args.split, vocabulary)
        elif method in {"d2", "d3"}:
            output = run_simple_llm_method(
                method,
                selected,
                args.split,
                vocabulary,
                vocab_text,
                args.force,
            )
        else:
            output = run_d4(
                selected,
                args.split,
                vocabulary,
                vocab_text,
                definitions,
                args.force,
            )
        selected_ids = {row["dynamic_case_id"] for row in selected}
        failures = [
            row
            for row in output
            if row.get("dynamic_case_id") in selected_ids
            and row.get("status") != "success"
        ]
        failed = failed or bool(failures)
        print(
            f"{method.upper()} rows={len(selected)} failures={len(failures)} "
            f"output={OUTPUT_PATHS[method](args.split)}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
