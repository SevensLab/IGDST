"""Shared data, model API and metric utilities for IGDST."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]


DATASET = os.getenv("IGDST_DATASET", "ihms")


DATA_DIR = Path(os.getenv("IGDST_DATA_DIR", str(ROOT.parent / "data")))
RESULTS_DIR = Path(os.getenv("IGDST_RESULTS_DIR", str(ROOT.parent / "results"))) / DATASET
STATIC_DIR = RESULTS_DIR / "static"
DYNAMIC_DIR = RESULTS_DIR / "dynamic"




DIMENSIONS = ["hazard", "operation_stage", "accident_type"]


GOLD_COLUMNS = {
    "hazard": "gold_hazard_tags",
    "operation_stage": "gold_operation_stage_tags",
    "accident_type": "gold_accident_type_tags",
}


PREDICTION_DIR = STATIC_DIR / "predictions"


METHODS = {
    "keyword": "Keyword",
    "embedding": "Embedding",
    "llm_only": "LLM-only",
    "llm_vocab": "LLM+Vocabulary",
    "igdst": "IGDST",
}


LAST_LLM_USAGE: dict[str, Any] | None = None


def ensure_output_dirs() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_env_file(path: Path | None = None) -> None:
    if path is not None:
        candidates = [path]
    else:
        candidates = [
            ROOT / ".env",
            ROOT / "config/.env",
        ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Keep explicit shell values, but allow a later fallback file to
            # replace an empty value from an earlier file.
            if key and not os.getenv(key) and value:
                os.environ[key] = value


def configured_value(value: str | None) -> bool:
    normalized = (value or "").strip()
    placeholders = {
        "",
        "...",
        "replace_with_your_api_key",
        "your_api_key_here",
        "sk-your-key-here",
    }
    return normalized not in placeholders


def llm_configured() -> bool:
    load_env_file()
    return bool(
        configured_value(os.getenv("LLM_API_KEY"))
        and configured_value(os.getenv("LLM_BASE_URL"))
        and configured_value(os.getenv("LLM_MODEL"))
    )


def call_llm_json(messages: list[dict[str, str]], temperature: float = 0, max_retries: int = 3) -> tuple[dict[str, Any], str]:
    global LAST_LLM_USAGE
    LAST_LLM_USAGE = None
    load_env_file()
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.environ["LLM_BASE_URL"].rstrip("/")
    model = os.environ["LLM_MODEL"]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    thinking = os.getenv("LLM_THINKING", "").strip()
    if thinking:
        payload["thinking"] = {"type": thinking}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = request.Request(f"{base_url}/chat/completions", data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            LAST_LLM_USAGE = normalize_llm_usage(data.get("usage"))
            raw = data["choices"][0]["message"]["content"]
            return extract_json_object(raw), raw
        except (error.URLError, error.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def normalize_llm_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    prompt = value.get("prompt_tokens", value.get("input_tokens"))
    completion = value.get("completion_tokens", value.get("output_tokens"))
    total = value.get("total_tokens")
    try:
        input_tokens = int(prompt or 0)
        output_tokens = int(completion or 0)
        total_tokens = int(total if total is not None else input_tokens + output_tokens)
    except (TypeError, ValueError):
        return None
    return {
        "api_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_estimated": False,
    }


def last_llm_usage() -> dict[str, Any] | None:
    return dict(LAST_LLM_USAGE) if LAST_LLM_USAGE else None


def llm_usage_or_estimate(system_prompt: str, user_prompt: str, raw: str = "") -> dict[str, Any]:
    actual = last_llm_usage()
    if actual:
        return actual
    input_tokens = max(1, math.ceil((len(system_prompt) + len(user_prompt)) / 4))
    output_tokens = math.ceil(len(raw) / 4) if raw else 0
    return {
        "api_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_estimated": True,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    return json.loads(match.group(0))


def find_gold_workbook(explicit: Path | None = None) -> Path:
    path = explicit or DATA_DIR / "gold" / f"{DATASET}.xlsx"
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not locate gold workbook: {path}")


def read_workbook(path: Path) -> dict[str, list[dict[str, str]]]:
    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Workbook reading requires openpyxl. Use your project Python environment "
            "or install openpyxl in the active environment."
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheets: dict[str, list[dict[str, str]]] = {}
    for sheet_name in workbook.sheetnames:
        ws = workbook[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            sheets[sheet_name] = []
            continue
        headers = [str(v).strip() if v is not None else "" for v in rows[0]]
        keep = [i for i, h in enumerate(headers) if h]
        sheet_rows = []
        for raw in rows[1:]:
            row = {}
            for i in keep:
                value = raw[i] if i < len(raw) else ""
                row[headers[i]] = cell_to_str(value)
            if any(v != "" for v in row.values()):
                sheet_rows.append(row)
        sheets[sheet_name] = sheet_rows
    workbook.close()
    return sheets


def cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def keyword_matches(text: str, pattern: str) -> bool:
    pattern = pattern.lower().strip()
    if not pattern:
        return False
    return pattern in text if " " in pattern or "-" in pattern else re.search(rf"\b{re.escape(pattern)}\b", text) is not None


def description_hash(description: str) -> str:
    return hashlib.sha256(normalize_text(description).encode("utf-8")).hexdigest()


def parse_multilabel(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def format_tag_set(tags: set[str] | list[str]) -> str:
    return ";".join(sorted(set(tags)))


def csv_read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_rows_by_pair(
    path: Path,
    expected_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in jsonl_read(path):
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            continue
        if expected_ids is not None and pair_id not in expected_ids:
            continue
        latest[pair_id] = row
    return latest


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def load_gold_data(workbook_path: Path | None = None) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, set[str]], list[dict[str, str]], Path]:
    path = find_gold_workbook(workbook_path)
    sheets = read_workbook(path)
    missing = {"Annotation", "Accident_Context", "Tag_Vocabulary"} - set(sheets)
    if missing:
        raise ValueError(f"Gold workbook missing sheets: {sorted(missing)}")
    annotations = sheets["Annotation"]
    contexts = {row["accident_id"]: row for row in sheets["Accident_Context"] if row.get("accident_id")}
    vocab_by_dimension: dict[str, set[str]] = {dim: set() for dim in DIMENSIONS}
    for row in sheets["Tag_Vocabulary"]:
        dim = row.get("dimension", "")
        tag = row.get("tag", "")
        if dim in vocab_by_dimension and tag:
            vocab_by_dimension[dim].add(tag)
    return annotations, contexts, vocab_by_dimension, sheets["Tag_Vocabulary"], path


def build_benchmark_pair(row: dict[str, str], context: dict[str, str]) -> dict[str, Any]:
    out = {
        "row_id": row.get("row_id", ""),
        "pair_id": row.get("pair_id", ""),
        "accident_id": row.get("accident_id", ""),
        "date": row.get("date", ""),
        "resource_id": row.get("resource_id", ""),
        "resource_name": row.get("resource_name", ""),
        "resource_type": row.get("resource_type", ""),
        "description": context.get("description", ""),
        "gold_include_resource": row.get("gold_include_resource", "").strip().lower(),
    }
    for dim in DIMENSIONS:
        out[f"gold_{dim}"] = format_tag_set(parse_multilabel(row.get(GOLD_COLUMNS[dim], "")))
    return out


def load_split_pairs(split: str) -> list[dict[str, str]]:
    if split not in {"dev", "test"}:
        raise ValueError("--split must be dev or test")
    return csv_read(STATIC_DIR / f"{split}_pairs.csv")


def predictions_from_jsonl(path: Path) -> dict[str, dict[str, set[str]]]:
    preds: dict[str, dict[str, set[str]]] = {}
    for row in latest_rows_by_pair(path).values():
        if row.get("status") and row.get("status") != "success":
            continue
        pred_obj = row.get("predictions", row)
        pair_preds: dict[str, set[str]] = {}
        for dim in DIMENSIONS:
            value = pred_obj.get(dim, [])
            if isinstance(value, str):
                pair_preds[dim] = parse_multilabel(value)
            else:
                pair_preds[dim] = {str(v).strip() for v in value if str(v).strip()}
        preds[row.get("pair_id", "")] = pair_preds
    return preds


def evaluate_predictions(pairs: list[dict[str, str]], predictions: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    by_dim: dict[str, Any] = {}
    label_rows = []
    for dim in DIMENSIONS:
        tp = fp = fn = 0
        jaccards = []
        empty_agree = empty_total = 0
        per_label_counts: dict[str, Counter] = defaultdict(Counter)
        positive_gold_count = 0
        positive_recall_hits = 0
        positive_jaccards = []
        positive_exact = 0
        for pair in pairs:
            gold = parse_multilabel(pair.get(f"gold_{dim}", ""))
            pred = predictions.get(pair["pair_id"], {}).get(dim, set())
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
            union = gold | pred
            if union:
                jaccards.append(len(gold & pred) / len(union))
            if not gold:
                empty_total += 1
                if not pred:
                    empty_agree += 1
            if gold:
                positive_gold_count += 1
                positive_recall_hits += len(gold & pred)
                positive_jaccards.append(len(gold & pred) / len(union) if union else 0.0)
                if gold == pred:
                    positive_exact += 1
            for tag in gold | pred:
                if tag in gold and tag in pred:
                    per_label_counts[tag]["tp"] += 1
                elif tag in pred:
                    per_label_counts[tag]["fp"] += 1
                elif tag in gold:
                    per_label_counts[tag]["fn"] += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        label_f1s = []
        for tag, c in sorted(per_label_counts.items()):
            lp = safe_div(c["tp"], c["tp"] + c["fp"])
            lr = safe_div(c["tp"], c["tp"] + c["fn"])
            lf = safe_div(2 * lp * lr, lp + lr)
            label_f1s.append(lf)
            label_rows.append(
                {
                    "dimension": dim,
                    "tag": tag,
                    "support": c["tp"] + c["fn"],
                    "precision": lp,
                    "recall": lr,
                    "f1": lf,
                    "rare_label": "RARE_LABEL" if (c["tp"] + c["fn"]) < 5 else "",
                }
            )
        by_dim[dim] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": mean(jaccards),
            "jaccard_n": len(jaccards),
            "exact_empty_agreement_rate": safe_div(empty_agree, empty_total),
            "empty_gold_count": empty_total,
            "label_macro_f1": mean(label_f1s),
            "positive_gold_count": positive_gold_count,
            "positive_recall": safe_div(positive_recall_hits, sum(len(parse_multilabel(p.get(f"gold_{dim}", ""))) for p in pairs if parse_multilabel(p.get(f"gold_{dim}", "")))),
            "positive_jaccard": mean(positive_jaccards),
            "positive_exact_match": safe_div(positive_exact, positive_gold_count),
        }
    overall = mean([by_dim[dim]["f1"] for dim in DIMENSIONS])
    return {"by_dimension": by_dim, "overall_macro_f1": overall, "per_label": label_rows}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def cluster_bootstrap(
    pairs: list[dict[str, str]],
    predictions: dict[str, dict[str, set[str]]],
    seed: int = 42,
    n_bootstrap: int = 2000,
) -> dict[str, tuple[float, float]]:
    by_accident: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        cluster_id = pair.get("accident_group_id") or pair["accident_id"]
        by_accident[cluster_id].append(pair)
    accident_ids = sorted(by_accident)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in ["overall_macro_f1", "hazard_f1", "operation_stage_f1", "accident_type_f1"]}
    for _ in range(n_bootstrap):
        drawn_pairs = []
        for _ in accident_ids:
            drawn_pairs.extend(by_accident[rng.choice(accident_ids)])
        metrics = evaluate_predictions(drawn_pairs, predictions)
        samples["overall_macro_f1"].append(metrics["overall_macro_f1"])
        for dim in DIMENSIONS:
            samples[f"{dim}_f1"].append(metrics["by_dimension"][dim]["f1"])
    return {name: (percentile(vals, 0.025), percentile(vals, 0.975)) for name, vals in samples.items()}


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""

def parse_date(value: str) -> date:
    return date.fromisoformat((value or "").strip()[:10])


def tag_set(row: dict[str, str], dimension: str, prefix: str = "gold_") -> set[str]:
    field = f"gold_{dimension}_tags" if prefix == "gold_" else f"{prefix}{dimension}_tags"
    return parse_multilabel(row.get(field, ""))


def load_dynamic_split(split: str) -> list[dict[str, str]]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'.")
    path = DYNAMIC_DIR / f"dynamic_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dynamic benchmark dynamic split: {path}")
    rows = csv_read(path)
    if not rows:
        raise ValueError(f"dynamic benchmark dynamic split is empty: {path}")
    return rows


def current_profile(row: dict[str, str]) -> dict[str, set[str]]:
    return {
        dimension: parse_multilabel(row.get(f"current_{dimension}_tags", ""))
        for dimension in DIMENSIONS
    }


def gold_add(row: dict[str, str]) -> dict[str, set[str]]:
    """Return evaluator-only Gold additions; prediction code must not call this."""
    return {
        dimension: parse_multilabel(row.get(f"gold_add_{dimension}_tags", ""))
        for dimension in DIMENSIONS
    }


def request_hash(
    method: str,
    model: str,
    temperature: float | int,
    system_prompt: str,
    exact_user_prompt: str,
) -> str:
    payload = method + model + str(temperature) + system_prompt + exact_user_prompt
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def span_exists(span: str, description: str) -> bool:
    span = str(span or "").strip()
    description = str(description or "")
    if not span:
        return False
    if span in description:
        return True
    normalized_span = re.sub(r"\s+", " ", span).strip()
    normalized_description = re.sub(r"\s+", " ", description).strip()
    return bool(normalized_span and normalized_span in normalized_description)


def normalize_prediction_tags(
    value: dict[str, Any],
    vocabulary: dict[str, set[str]],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for dimension in DIMENSIONS:
        raw_items = value.get(dimension, []) if isinstance(value, dict) else []
        if isinstance(raw_items, str):
            items: list[Any] = [raw_items]
        elif isinstance(raw_items, (list, tuple, set)):
            items = list(raw_items)
        else:
            items = []
        tags = []
        for item in items:
            if isinstance(item, dict):
                item = item.get("tag", "")
            original = str(item or "").strip()
            if not original:
                continue
            tags.append(
                original
                if original in vocabulary.get(dimension, set())
                else f"UNMAPPED::{original}"
            )
        normalized[dimension] = sorted(set(tags))
    return normalized


def gold_validation_failures(
    annotations: list[dict[str, str]],
    contexts: dict[str, dict[str, str]],
    vocab: dict[str, set[str]],
) -> list[str]:
    failures = []
    pair_ids = [row.get("pair_id", "").strip() for row in annotations]
    if not pair_ids or any(not pair_id for pair_id in pair_ids):
        failures.append("blank pair_id")
    if len(pair_ids) != len(set(pair_ids)):
        failures.append("duplicate pair_id")
    for row in annotations:
        include = row.get("gold_include_resource", "").strip().lower()
        if include not in {"yes", "no"}:
            failures.append("invalid gold_include_resource")
            break
        if not row.get("resource_id", "").strip():
            failures.append("blank resource_id")
            break
        if row.get("accident_id", "").strip() not in contexts:
            failures.append("missing accident context")
            break
        if row.get("resource_type", "").strip() not in {"chemical", "equipment", "consumable"}:
            failures.append("invalid resource_type")
            break
        for dimension in DIMENSIONS:
            invalid = parse_multilabel(row.get(f"gold_{dimension}_tags", "")) - vocab.get(dimension, set())
            if invalid:
                failures.append(f"invalid {dimension} tag")
                return failures
    return failures


def load_approved_gold_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, set[str]], list[dict[str, str]], Path]:
    annotations, contexts, vocab, vocab_rows, workbook_path = load_gold_data()
    failures = gold_validation_failures(annotations, contexts, vocab)
    if failures:
        raise RuntimeError("Frozen static benchmark Gold source is unavailable: " + "; ".join(failures))
    rows = [
        row
        for row in annotations
        if row.get("gold_include_resource", "").strip().lower() == "yes"
    ]
    return rows, contexts, vocab, vocab_rows, workbook_path


def compact_vocab_text(vocab: dict[str, set[str]], vocab_rows: list[dict[str, str]]) -> str:
    definitions = {(row.get("dimension", ""), row.get("tag", "")): row.get("definition", "") for row in vocab_rows}
    chunks = []
    for dimension in DIMENSIONS:
        items = []
        for tag in sorted(vocab.get(dimension, set())):
            definition = normalize_text(definitions.get((dimension, tag), ""))
            if len(definition) > 120:
                definition = definition[:117].rstrip() + "..."
            items.append(f"{tag}: {definition}")
        chunks.append(f"{dimension}: " + " | ".join(items))
    return "\n".join(chunks)


def enrich_gold_rows():
    """Build temporal inputs directly from the retained reference workbook."""
    rows, contexts, vocab, vocab_rows, workbook_path = load_approved_gold_rows()
    hashes = {key: description_hash(ctx.get("description", "")) for key, ctx in contexts.items()}
    counts = Counter(hashes.values())
    duplicate_ids = {}
    for h in hashes.values():
        if counts[h] > 1 and h not in duplicate_ids:
            duplicate_ids[h] = f"DUP_{len(duplicate_ids)+1:04d}"
    enriched, seen = [], set()
    for row in rows:
        ctx = contexts[row["accident_id"]]
        key = (row["resource_id"], row["accident_id"])
        if key in seen:
            raise ValueError(f"Duplicate resource–accident pair: {key}")
        seen.add(key)
        description = ctx.get("description", "").strip()
        if not description:
            raise ValueError(f"Missing description: {row['accident_id']}")
        h = hashes[row["accident_id"]]
        duplicate = duplicate_ids.get(h, "")
        enriched.append({**row, "date": parse_date(row.get("date") or ctx["date"]).isoformat(),
                         "description": description, "description_hash": h,
                         "duplicate_group_id": duplicate, "accident_group_id": duplicate or h})
    return enriched, contexts, {}, vocab, vocab_rows, workbook_path
