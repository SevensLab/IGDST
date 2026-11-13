"""Static tagging with keyword, embedding, language-model and IGDST methods."""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Any
import numpy as np
from common import (DIMENSIONS, ROOT, STATIC_DIR, PREDICTION_DIR, ensure_output_dirs,
                    call_llm_json, json_read, json_write, jsonl_read, jsonl_write,
                    llm_configured, llm_usage_or_estimate, now_utc, load_split_pairs,
                    request_hash, keyword_matches, evaluate_predictions)

def select_pairs(split: str, limit: int | None = None, pair_ids: set[str] | None = None) -> list[dict[str, str]]:
    pairs = load_split_pairs(split)
    if pair_ids:
        pairs = [pair for pair in pairs if pair["pair_id"] in pair_ids]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def vocabulary_text(tag_defs: dict[str, dict[str, str]]) -> str:
    parts = []
    parts.append(
        "Vocabulary policy: labels are evidence-based and resource-specific. "
        "An empty array is valid and preferred when the accident text does not "
        "provide affirmative evidence for a dimension. Do not infer a label from "
        "resource presence, name, type, location, PPE, emergency equipment, medical "
        "response, or another resource in the same accident."
    )
    for dim in DIMENSIONS:
        parts.append(f"Allowed {dim} tags:")
        for tag, definition in sorted(tag_defs[dim].items()):
            parts.append(f"- {tag}: {definition}")
    return "\n".join(parts)


def pair_input_text(pair: dict[str, str]) -> str:
    return (
        f"Resource name: {pair['resource_name']}\n"
        f"Resource type: {pair['resource_type']}\n"
        f"Accident description:\n{pair['description']}"
    )


def normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    out = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("tag", "")
        item = str(item).strip()
        if item:
            out.append(item)
    return sorted(set(out))


def normalize_controlled_output(obj: dict[str, Any], allowed: dict[str, set[str]]) -> dict[str, list[str]]:
    preds = {}
    for dim in DIMENSIONS:
        tags = []
        for tag in normalize_list(obj.get(dim, [])):
            if tag in allowed[dim]:
                tags.append(tag)
            else:
                tags.append(f"UNMAPPED::{tag}")
        preds[dim] = sorted(set(tags))
    return preds


def canonicalize_free_output(obj: dict[str, Any], synonym_map: dict[str, dict[str, str]], allowed: dict[str, set[str]]) -> dict[str, list[str]]:
    preds = {}
    for dim in DIMENSIONS:
        normalized_allowed = {
            normalize_free_label(canonical_tag): canonical_tag
            for canonical_tag in allowed[dim]
        }
        normalized_synonyms = {
            normalize_free_label(source): canonical_tag
            for source, canonical_tag in synonym_map.get(dim, {}).items()
        }
        tags = []
        for item in normalize_list(obj.get(dim, [])):
            key = normalize_free_label(item)
            if key in normalized_allowed:
                tags.append(normalized_allowed[key])
            elif key in normalized_synonyms:
                tags.append(normalized_synonyms[key])
            else:
                tags.append(f"UNMAPPED::{item}")
        preds[dim] = sorted(set(tags))
    return preds


def normalize_free_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_candidates(obj: dict[str, Any], allowed: dict[str, set[str]]) -> dict[str, list[dict[str, str]]]:
    out = {dim: [] for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        value = obj.get(dim, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            tag = str(item.get("tag", "")).strip()
            if not tag:
                continue
            if tag not in allowed[dim]:
                tag = f"UNMAPPED::{tag}"
            out[dim].append(
                {
                    "tag": tag,
                    "evidence_span": str(item.get("evidence_span", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
    return out


def extract_free_candidates(
    obj: dict[str, Any],
    synonym_map: dict[str, dict[str, str]],
    allowed: dict[str, set[str]],
) -> dict[str, list[dict[str, str]]]:
    out = {dim: [] for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        value = obj.get(dim, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            original_tag = str(item.get("tag", "")).strip()
            if not original_tag:
                continue
            canonical = canonicalize_free_output(
                {dim: [original_tag]}, synonym_map, allowed
            )[dim][0]
            out[dim].append(
                {
                    "tag": canonical,
                    "evidence_span": str(item.get("evidence_span", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
    return out


def run_llm_method(
    *,
    method: str,
    prompt_path: Path,
    output_path: Path,
    split: str,
    limit: int | None,
    pair_ids: set[str] | None,
    force: bool,
    prompt_builder,
    response_parser,
) -> list[dict[str, Any]]:
    if not llm_configured():
        raise RuntimeError("Missing LLM_API_KEY, LLM_BASE_URL, or LLM_MODEL. LLM predictions were not run.")
    pairs = select_pairs(split, limit, pair_ids)
    split_pair_ids = {pair["pair_id"] for pair in load_split_pairs(split)}
    prompt = prompt_path.read_text(encoding="utf-8")
    rows = (
        []
        if force
        else [
            row
            for row in jsonl_read(output_path)
            if row.get("pair_id") in split_pair_ids
        ]
    )
    model = os.getenv("LLM_MODEL", "")
    for pair in pairs:
        user_prompt = prompt_builder(pair)
        current_hash = request_hash(method, model, 0, prompt, user_prompt)
        cached = [
            row
            for row in rows
            if row.get("pair_id") == pair["pair_id"]
            and row.get("request_hash") == current_hash
            and row.get("status") == "success"
        ]
        if not force and cached:
            rows = [row for row in rows if row.get("pair_id") != pair["pair_id"]]
            rows.append(cached[-1])
            jsonl_write(output_path, rows)
            continue
        rows = [row for row in rows if row.get("pair_id") != pair["pair_id"]]
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ]
            parsed, raw = call_llm_json(messages, temperature=0, max_retries=3)
            record = response_parser(pair, parsed, raw)
            record.update(llm_usage_or_estimate(prompt, user_prompt, raw))
        except Exception as exc:
            record = {
                "pair_id": pair["pair_id"],
                "accident_id": pair["accident_id"],
                "resource_id": pair["resource_id"],
                "method": method,
                "status": "failed",
                "failure_reason": str(exc),
                "timestamp": now_utc(),
                **llm_usage_or_estimate(prompt, user_prompt),
            }
        record["request_hash"] = current_hash
        rows.append(record)
        jsonl_write(output_path, rows)
    return rows

def variant_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_incident:
        return {
            "method": "ablation_no_incident",
            "stem": "ablation_no_incident",
            "prompt_name": "llm_vocab_v1.txt",
            "uses_evidence": False,
            "uses_vocab": True,
            "uses_resource": True,
            "uses_verifier": False,
            "generator_cache_method": "ablation_no_incident",
        }
    if args.no_vocab:
        return {
            "method": "ablation_no_vocab",
            "stem": "ablation_no_vocab",
            "prompt_name": "igdst_generator_v1.txt",
            "uses_evidence": True,
            "uses_vocab": False,
            "uses_resource": True,
            "uses_verifier": False,
            "candidate_stem": "ablation_no_vocab",
            "filter_stem": "ablation_no_vocab_evidence_filter",
            "generator_cache_method": "ablation_no_vocab",
        }
    if args.no_resource_context:
        return {
            "method": "ablation_no_resource_context",
            "stem": "ablation_no_resource_context",
            "prompt_name": "igdst_generator_v1.txt",
            "uses_evidence": True,
            "uses_vocab": True,
            "uses_resource": False,
            "uses_verifier": False,
            "candidate_stem": "ablation_no_resource_context",
            "filter_stem": "ablation_no_resource_context_evidence_filter",
            "generator_cache_method": "ablation_no_resource_context",
        }
    if args.hard_verifier:
        return {
            "method": "igdst_hard_verifier",
            "stem": "igdst_hardvrf",
            "prompt_name": "igdst_generator_v1.txt",
            "uses_evidence": True,
            "uses_vocab": True,
            "uses_resource": True,
            "uses_verifier": True,
            # The generator is identical to Full IGDST, so reuse its frozen cache.
            "candidate_stem": "igdst",
            "filter_stem": "igdst",
            "generator_cache_method": "igdst",
            "verifier_cache_method": "igdst",
        }
    return {
        "method": "igdst",
        "stem": "igdst",
        "prompt_name": "igdst_generator_v1.txt",
        "uses_evidence": True,
        "uses_vocab": True,
        "uses_resource": True,
        # Static IGDST uses deterministic canonical/evidence filtering only.
        "uses_verifier": False,
        "candidate_stem": "igdst",
        "filter_stem": "igdst_evidence_filter",
        "generator_cache_method": "igdst",
    }


def generator_user_prompt(pair: dict[str, str], settings: dict[str, Any], vocab_block: str) -> str:
    if not settings["uses_evidence"]:
        text = (
            f"Resource name: {pair['resource_name']}\n"
            f"Resource type: {pair['resource_type']}"
        )
    elif not settings["uses_resource"]:
        text = f"Accident description:\n{pair['description']}"
    else:
        text = pair_input_text(pair)
    if settings["uses_vocab"]:
        text += "\n\n" + vocab_block
    return text


def verifier_user_prompt(
    pair: dict[str, str],
    dimension: str,
    candidate: dict[str, str],
    definition: str,
    include_resource: bool,
) -> str:
    parts = []
    if include_resource:
        parts.extend(
            [
                f"Resource name: {pair['resource_name']}",
                f"Resource type: {pair['resource_type']}",
            ]
        )
    parts.extend(
        [
            f"Accident description:\n{pair['description']}",
            f"Predicted dimension: {dimension}",
            f"Predicted tag: {candidate['tag']}",
            f"Tag definition: {definition}",
            f"Evidence span: {candidate.get('evidence_span', '')}",
        ]
    )
    return "\n\n".join(parts)


def sanitize_candidate_evidence(
    candidates: dict[str, list[dict[str, str]]],
    description: str,
) -> dict[str, list[dict[str, str]]]:
    """Keep only verbatim spans while preserving rejected model text for QA."""
    for dimension in DIMENSIONS:
        for candidate in candidates.get(dimension, []):
            span = str(candidate.get("evidence_span", "")).strip()
            if not span or span in description:
                continue
            candidate["raw_evidence_span"] = span
            match = re.search(re.escape(span), description, flags=re.IGNORECASE)
            if match:
                candidate["evidence_span"] = match.group(0)
                candidate["evidence_normalization"] = "case_restored_from_description"
            else:
                candidate["evidence_span"] = ""
                candidate["evidence_normalization"] = "nonverbatim_span_cleared"
    return candidates


def deterministic_static_filter(
    tag: str, evidence_span: str, description: str
) -> tuple[str, str]:
    """Apply the non-semantic static canonical/exact-evidence gate."""
    if tag.startswith("UNMAPPED::"):
        return (
            "UNSUPPORTED",
            "Static IGDST v2: deterministic canonical filter rejects UNMAPPED tags.",
        )
    if not evidence_span or evidence_span not in description:
        return "UNSUPPORTED", "Evidence span does not occur in description."
    return (
        "SUPPORTED",
        "Static IGDST v2: canonical tag with an exact verbatim evidence span.",
    )


def run_no_incident(
    pairs: list[dict[str, str]],
    settings: dict[str, Any],
    final_path: Path,
    system_prompt: str,
    vocab_block: str,
    vocab: dict[str, set[str]],
    force: bool,
) -> int:
    rows = [] if force else jsonl_read(final_path)
    model = os.getenv("LLM_MODEL", "")
    for pair in pairs:
        user_prompt = generator_user_prompt(pair, settings, vocab_block)
        current_hash = request_hash(settings["method"], model, 0, system_prompt, user_prompt)
        cached = [
            row
            for row in rows
            if row.get("pair_id") == pair["pair_id"]
            and row.get("request_hash") == current_hash
            and row.get("status") == "success"
        ]
        rows = [row for row in rows if row.get("pair_id") != pair["pair_id"]]
        if cached and not force:
            rows.append(cached[-1])
            continue
        try:
            parsed, raw = call_llm_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_retries=3,
            )
            usage = llm_usage_or_estimate(system_prompt, user_prompt, raw)
            record = {
                "pair_id": pair["pair_id"],
                "accident_id": pair["accident_id"],
                "resource_id": pair["resource_id"],
                "method": settings["method"],
                "prompt_version": "llm_vocab_v1",
                "model": model,
                "predictions": normalize_controlled_output(parsed, vocab),
                "raw_response": raw,
                "status": "success",
                "failure_reason": "",
                "request_hash": current_hash,
                "timestamp": now_utc(),
                **usage,
            }
        except Exception as exc:
            record = {
                "pair_id": pair["pair_id"],
                "accident_id": pair["accident_id"],
                "resource_id": pair["resource_id"],
                "method": settings["method"],
                "model": model,
                "status": "failed",
                "failure_reason": f"generator_failed:{exc}",
                "request_hash": current_hash,
                "timestamp": now_utc(),
                **llm_usage_or_estimate(system_prompt, user_prompt),
            }
        rows.append(record)
        jsonl_write(final_path, rows)
    jsonl_write(final_path, rows)
    return len(rows)


def run_igdst(args) -> int:
    ensure_output_dirs()
    if not llm_configured():
        raise RuntimeError("Missing LLM_API_KEY, LLM_BASE_URL, or LLM_MODEL. IGDST predictions were not run.")

    settings = variant_settings(args)
    tag_defs = json_read(ROOT / "config/tag_definitions_en_v1.json")
    vocab = {dim: set(tags) for dim, tags in tag_defs.items()}
    synonym_map = json_read(ROOT / "config/llm_only_mapping_v1.json")
    vocab_block = vocabulary_text(tag_defs)
    pair_ids = {value.strip() for value in args.pair_ids.split(",") if value.strip()} or None
    pairs = select_pairs(args.split, args.limit, pair_ids)
    split_pair_ids = {
        pair["pair_id"] for pair in select_pairs(args.split, None, None)
    }
    model = os.getenv("LLM_MODEL", "")

    final_path = PREDICTION_DIR / f"{settings['stem']}_{args.split}.jsonl"
    generator_prompt = (ROOT / "prompts" / settings["prompt_name"]).read_text(encoding="utf-8")
    if not settings["uses_evidence"]:
        count = run_no_incident(
            pairs,
            settings,
            final_path,
            generator_prompt,
            vocab_block,
            vocab,
            args.force,
        )
        print(f"Wrote {count} {settings['method']} predictions to {final_path}")
        return 0

    candidate_stem = settings.get("candidate_stem", settings["stem"])
    filter_stem = settings.get("filter_stem", settings["stem"])
    candidates_path = PREDICTION_DIR / f"{candidate_stem}_candidates_{args.split}.jsonl"
    verifier_path = (
        PREDICTION_DIR / f"{filter_stem}_verifier_{args.split}.jsonl"
        if settings["uses_verifier"]
        else PREDICTION_DIR / f"{filter_stem}_{args.split}.jsonl"
    )
    verifier_prompt = (ROOT / "prompts/evidence_verifier_v1.txt").read_text(encoding="utf-8")

    candidate_rows = (
        []
        if args.force
        else [
            row
            for row in jsonl_read(candidates_path)
            if row.get("pair_id") in split_pair_ids
        ]
    )
    current_candidates: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        user_prompt = generator_user_prompt(pair, settings, vocab_block)
        current_hash = request_hash(
            settings.get("generator_cache_method", settings["method"]),
            model,
            0,
            generator_prompt,
            user_prompt,
        )
        cached = [
            row
            for row in candidate_rows
            if row.get("pair_id") == pair["pair_id"]
            and row.get("request_hash") == current_hash
            and row.get("status") == "success"
        ]
        candidate_rows = [row for row in candidate_rows if row.get("pair_id") != pair["pair_id"]]
        if cached and not args.force:
            record = cached[-1]
        else:
            try:
                parsed, raw = call_llm_json(
                    [
                        {"role": "system", "content": generator_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_retries=3,
                )
                usage = llm_usage_or_estimate(generator_prompt, user_prompt, raw)
                if settings["uses_vocab"]:
                    candidates = extract_candidates(parsed, vocab)
                else:
                    candidates = extract_free_candidates(parsed, synonym_map, vocab)
                record = {
                    "pair_id": pair["pair_id"],
                    "accident_id": pair["accident_id"],
                    "resource_id": pair["resource_id"],
                    "method": settings["method"],
                    "prompt_version": "igdst_generator_v1",
                    "model": model,
                    "candidates": candidates,
                    "raw_response": raw,
                    "status": "success",
                    "failure_reason": "",
                    "request_hash": current_hash,
                    "timestamp": now_utc(),
                    **usage,
                }
            except Exception as exc:
                record = {
                    "pair_id": pair["pair_id"],
                    "accident_id": pair["accident_id"],
                    "resource_id": pair["resource_id"],
                    "method": settings["method"],
                    "model": model,
                    "status": "failed",
                    "failure_reason": str(exc),
                    "request_hash": current_hash,
                    "timestamp": now_utc(),
                    **llm_usage_or_estimate(generator_prompt, user_prompt),
                }
        if record.get("status") == "success":
            record["candidates"] = sanitize_candidate_evidence(
                record.get("candidates", {}), pair["description"]
            )
        candidate_rows.append(record)
        current_candidates[pair["pair_id"]] = record
        jsonl_write(candidates_path, candidate_rows)

    old_verifier_rows = (
        []
        if args.force
        else [
            row
            for row in jsonl_read(verifier_path)
            if row.get("pair_id") in split_pair_ids
        ]
    )
    selected_ids = {pair["pair_id"] for pair in pairs}
    verifier_rows = [row for row in old_verifier_rows if row.get("pair_id") not in selected_ids]
    current_verifiers: dict[str, list[dict[str, Any]]] = {pair_id: [] for pair_id in selected_ids}
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}

    for pair_id, candidate_record in current_candidates.items():
        if candidate_record.get("status") != "success":
            continue
        pair = pair_by_id[pair_id]
        for dimension in DIMENSIONS:
            for candidate in candidate_record.get("candidates", {}).get(dimension, []):
                span = candidate.get("evidence_span", "")
                definition = tag_defs[dimension].get(candidate["tag"], "")
                verifier_input = verifier_user_prompt(
                    pair,
                    dimension,
                    candidate,
                    definition,
                    settings["uses_resource"],
                )
                filter_method = (
                    f"{settings.get('verifier_cache_method', settings['method'])}:verifier"
                    if settings["uses_verifier"]
                    else f"{settings['method']}:deterministic_evidence_v2"
                )
                current_hash = request_hash(
                    filter_method,
                    model,
                    0,
                    verifier_prompt,
                    verifier_input,
                )
                cached = [
                    row
                    for row in old_verifier_rows
                    if row.get("pair_id") == pair_id
                    and row.get("request_hash") == current_hash
                    and row.get("status") == "success"
                ]
                if cached and not args.force:
                    verifier_record = cached[-1]
                else:
                    span_exists = bool(span) and span in pair["description"]
                    base_record = {
                        "pair_id": pair_id,
                        "accident_id": pair["accident_id"],
                        "resource_id": pair["resource_id"],
                        "dimension": dimension,
                        "tag": candidate["tag"],
                        "evidence_span": span,
                        "span_exists": span_exists,
                        "request_hash": current_hash,
                        "timestamp": now_utc(),
                    }
                    if not settings["uses_verifier"]:
                        verdict, rationale = deterministic_static_filter(
                            candidate["tag"], span, pair["description"]
                        )
                        verifier_record = {
                            **base_record,
                            "verifier_result": verdict,
                            "rationale": rationale,
                            "filter_kind": "deterministic_canonical_evidence",
                            "raw_response": "",
                            "status": "success",
                            "failure_reason": "",
                        }
                    elif not span_exists:
                        verifier_record = {
                            **base_record,
                            "verifier_result": "UNSUPPORTED",
                            "rationale": "Evidence span does not occur in description.",
                            "raw_response": "",
                            "status": "success",
                            "failure_reason": "",
                        }
                    elif candidate["tag"].startswith("UNMAPPED::"):
                        verifier_record = {
                            **base_record,
                            "verifier_result": "UNSUPPORTED",
                            "rationale": "UNMAPPED tag has no frozen definition and is not sent to the verifier API.",
                            "raw_response": "",
                            "status": "success",
                            "failure_reason": "",
                        }
                    else:
                        try:
                            parsed, raw = call_llm_json(
                                [
                                    {"role": "system", "content": verifier_prompt},
                                    {"role": "user", "content": verifier_input},
                                ],
                                temperature=0,
                                max_retries=3,
                            )
                            usage = llm_usage_or_estimate(verifier_prompt, verifier_input, raw)
                            verdict = str(parsed.get("verdict", "")).strip().upper()
                            if verdict not in {"SUPPORTED", "UNSUPPORTED"}:
                                raise ValueError(f"Invalid verifier verdict: {verdict or '<blank>'}")
                            verifier_record = {
                                **base_record,
                                "verifier_result": verdict,
                                "rationale": str(parsed.get("rationale", "")).strip(),
                                "raw_response": raw,
                                "status": "success",
                                "failure_reason": "",
                                **usage,
                            }
                        except Exception as exc:
                            verifier_record = {
                                **base_record,
                                "verifier_result": "",
                                "rationale": "",
                                "raw_response": "",
                                "status": "failed",
                                "failure_reason": str(exc),
                                **llm_usage_or_estimate(verifier_prompt, verifier_input),
                            }
                verifier_rows.append(verifier_record)
                current_verifiers[pair_id].append(verifier_record)
                jsonl_write(verifier_path, verifier_rows)
    jsonl_write(verifier_path, verifier_rows)

    existing_final = (
        []
        if args.force
        else [
            row
            for row in jsonl_read(final_path)
            if row.get("pair_id") in split_pair_ids
        ]
    )
    final_rows = [row for row in existing_final if row.get("pair_id") not in selected_ids]
    for pair in pairs:
        candidate_record = current_candidates.get(pair["pair_id"])
        base_final = {
            "pair_id": pair["pair_id"],
            "accident_id": pair["accident_id"],
            "resource_id": pair["resource_id"],
            "method": settings["method"],
            "prompt_version": (
                "igdst_generator_v1+legacy_hard_verifier_v1"
                if settings["uses_verifier"]
                else "igdst_generator_v1+deterministic_evidence_filter_v2"
            ),
            "model": model,
            "request_hash": candidate_record.get("request_hash", "") if candidate_record else "",
            "timestamp": now_utc(),
        }
        if not candidate_record or candidate_record.get("status") != "success":
            reason = candidate_record.get("failure_reason", "missing generator result") if candidate_record else "missing generator result"
            final_rows.append(
                {
                    **base_final,
                    "status": "failed",
                    "failure_reason": f"generator_failed:{reason}",
                }
            )
            continue

        expected_count = sum(
            len(candidate_record.get("candidates", {}).get(dimension, []))
            for dimension in DIMENSIONS
        )
        verifier_records = current_verifiers.get(pair["pair_id"], [])
        verifier_failures = [
            row.get("failure_reason", "unknown verifier failure")
            for row in verifier_records
            if row.get("status") != "success"
        ]
        if len(verifier_records) != expected_count:
            verifier_failures.append(
                f"missing_verifier_results:{len(verifier_records)}/{expected_count}"
            )
        if verifier_failures:
            final_rows.append(
                {
                    **base_final,
                    "status": "failed",
                    "failure_reason": "verifier_failed:" + " | ".join(verifier_failures),
                }
            )
            continue

        predictions = {dimension: [] for dimension in DIMENSIONS}
        for row in verifier_records:
            if row.get("span_exists") and row.get("verifier_result") == "SUPPORTED":
                predictions[row["dimension"]].append(row["tag"])
        final_rows.append(
            {
                **base_final,
                "predictions": {
                    dimension: sorted(set(predictions[dimension]))
                    for dimension in DIMENSIONS
                },
                "status": "success",
                "failure_reason": "",
                "api_calls": int(candidate_record.get("api_calls", 0))
                + sum(int(row.get("api_calls", 0)) for row in verifier_records),
                "input_tokens": int(candidate_record.get("input_tokens", 0))
                + sum(int(row.get("input_tokens", 0)) for row in verifier_records),
                "output_tokens": int(candidate_record.get("output_tokens", 0))
                + sum(int(row.get("output_tokens", 0)) for row in verifier_records),
                "total_tokens": int(candidate_record.get("total_tokens", 0))
                + sum(int(row.get("total_tokens", 0)) for row in verifier_records),
                "usage_estimated": bool(candidate_record.get("usage_estimated", False))
                or any(bool(row.get("usage_estimated", False)) for row in verifier_records),
            }
        )

    jsonl_write(final_path, final_rows)
    print(f"Wrote {len(final_rows)} {settings['method']} predictions to {final_path}")
    return 0

def keyword_predict(pair: dict[str, str], rules: dict) -> dict[str, list[str]]:
    text = f"{pair.get('resource_name', '')} {pair.get('description', '')}".lower()
    predictions: dict[str, list[str]] = {dim: [] for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        for tag, patterns in rules.get(dim, {}).items():
            for pattern in patterns:
                if keyword_matches(text, pattern):
                    predictions[dim].append(tag)
                    break
    return {dim: sorted(set(predictions[dim])) for dim in DIMENSIONS}


def run_keyword(split: str, rules_path: Path) -> list[dict]:
    rules = json_read(rules_path)
    pairs = load_split_pairs(split)
    rows = []
    for pair in pairs:
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "accident_id": pair["accident_id"],
                "resource_id": pair["resource_id"],
                "method": "keyword",
                "prompt_version": "keyword_rules_v1",
                "model": "none",
                "predictions": keyword_predict(pair, rules),
                "status": "success",
                "failure_reason": "",
                "timestamp": now_utc(),
            }
        )
    return rows

DEFAULT_GRID = [round(x / 100, 2) for x in range(20, 81, 5)]


def require_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Embedding baseline requires sentence-transformers and the fixed model "
            f"{model_name!r}. Install/provide it before running M2; no fallback model is used."
        ) from exc
    return SentenceTransformer(model_name)


def encode_normalized(model, texts: list[str]) -> np.ndarray:
    emb = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    return emb


def pair_text(pair: dict[str, str]) -> str:
    return f"Resource: {pair['resource_name']}\nResource type: {pair['resource_type']}\nAccident: {pair['description']}"


def score_pairs(model, pairs: list[dict[str, str]], tag_defs: dict) -> dict[str, dict[str, dict[str, float]]]:
    pair_embeddings = encode_normalized(model, [pair_text(pair) for pair in pairs])
    scores: dict[str, dict[str, dict[str, float]]] = {pair["pair_id"]: {dim: {} for dim in DIMENSIONS} for pair in pairs}
    for dim in DIMENSIONS:
        tags = sorted(tag_defs[dim])
        tag_embeddings = encode_normalized(model, [tag_defs[dim][tag] for tag in tags])
        sim = pair_embeddings @ tag_embeddings.T
        for i, pair in enumerate(pairs):
            scores[pair["pair_id"]][dim] = {tag: float(sim[i, j]) for j, tag in enumerate(tags)}
    return scores


def predictions_at_threshold(scores: dict, thresholds: dict[str, float]) -> dict[str, dict[str, set[str]]]:
    preds = {}
    for pair_id, by_dim in scores.items():
        preds[pair_id] = {}
        for dim in DIMENSIONS:
            threshold = thresholds[dim]
            preds[pair_id][dim] = {tag for tag, score in by_dim[dim].items() if score >= threshold}
    return preds


def tune_thresholds(dev_pairs: list[dict[str, str]], dev_scores: dict, grid: list[float]) -> dict[str, float]:
    thresholds = {}
    for dim in DIMENSIONS:
        best_threshold = grid[0]
        best_f1 = -1.0
        for threshold in grid:
            preds = predictions_at_threshold(dev_scores, {d: threshold for d in DIMENSIONS})
            metrics = evaluate_predictions(dev_pairs, preds)
            f1 = metrics["by_dimension"][dim]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[dim] = best_threshold
    return thresholds


def run_embedding(split: str, model_name: str, thresholds_path: Path, tag_defs_path: Path) -> list[dict]:
    model = require_model(model_name)
    tag_defs = json_read(tag_defs_path)
    pairs = load_split_pairs(split)
    scores = score_pairs(model, pairs, tag_defs)
    if split == "dev":
        thresholds = tune_thresholds(pairs, scores, DEFAULT_GRID)
        json_write(thresholds_path, {"model": model_name, "grid": DEFAULT_GRID, "thresholds": thresholds})
    else:
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Missing frozen embedding thresholds: {thresholds_path}")
        thresholds = json_read(thresholds_path)["thresholds"]
    preds = predictions_at_threshold(scores, thresholds)
    rows = []
    for pair in pairs:
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "accident_id": pair["accident_id"],
                "resource_id": pair["resource_id"],
                "method": "embedding",
                "prompt_version": "embedding_retrieval_v1",
                "model": model_name,
                "thresholds": thresholds,
                "scores": scores[pair["pair_id"]],
                "predictions": {dim: sorted(preds[pair["pair_id"]][dim]) for dim in DIMENSIONS},
                "status": "success",
                "failure_reason": "",
                "timestamp": now_utc(),
            }
        )
    return rows


def run_baseline_llm(args):
    tag_defs = json_read(ROOT / "config/tag_definitions_en_v1.json")
    vocab = {dim: set(tags) for dim, tags in tag_defs.items()}
    synonyms = json_read(ROOT / "config/llm_only_mapping_v1.json")
    pair_ids = set(filter(None, (x.strip() for x in args.pair_ids.split(",")))) or None
    def build(pair):
        text = pair_input_text(pair)
        return text + "\n\n" + vocabulary_text(tag_defs) if args.method == "llm_vocab" else text
    def parse(pair, parsed, raw):
        predictions = (normalize_controlled_output(parsed, vocab) if args.method == "llm_vocab"
                       else canonicalize_free_output(parsed, synonyms, vocab))
        return {"pair_id": pair["pair_id"], "accident_id": pair["accident_id"],
                "resource_id": pair["resource_id"], "method": args.method,
                "prompt_version": args.method + "_v1", "model": os.getenv("LLM_MODEL", ""),
                "predictions": predictions, "raw_response": raw, "status": "success",
                "failure_reason": "", "timestamp": now_utc()}
    return run_llm_method(method=args.method, prompt_path=ROOT / "prompts" / f"{args.method}_v1.txt",
                          output_path=PREDICTION_DIR / f"{args.method}_{args.split}.jsonl",
                          split=args.split, limit=args.limit, pair_ids=pair_ids, force=args.force,
                          prompt_builder=build, response_parser=parse)


def main():
    parser = argparse.ArgumentParser(description="Run static tagging methods and IGDST ablations.")
    parser.add_argument("--method", choices=["keyword", "embedding", "llm_only", "llm_vocab", "igdst"], default="igdst")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pair-ids", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--thresholds", type=Path, default=STATIC_DIR / "embedding_thresholds_v1.json")
    variants = parser.add_mutually_exclusive_group()
    for flag in ["hard-verifier", "no-incident", "no-vocab", "no-resource-context"]:
        variants.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    if args.method != "igdst" and any([args.hard_verifier, args.no_incident, args.no_vocab, args.no_resource_context]):
        parser.error("Ablation switches apply only to --method igdst")
    if args.method in {"keyword", "embedding"} and (args.limit is not None or args.pair_ids):
        parser.error("Keyword and embedding baselines run the complete split")
    ensure_output_dirs()
    if args.method == "igdst":
        return run_igdst(args)
    if args.method == "keyword":
        rows = run_keyword(args.split, ROOT / "config/keyword_rules_v1.json")
    elif args.method == "embedding":
        rows = run_embedding(args.split, args.model, args.thresholds, ROOT / "config/tag_definitions_en_v1.json")
    else:
        rows = run_baseline_llm(args)
    out = PREDICTION_DIR / f"{args.method}_{args.split}.jsonl"
    jsonl_write(out, rows)
    print(f"{args.method}: {len(rows)} predictions in {out}")
    return 0 if all(row.get("status") == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
