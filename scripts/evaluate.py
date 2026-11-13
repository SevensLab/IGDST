"""Static and dynamic metrics with accident-cluster bootstrap intervals."""
from __future__ import annotations
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from common import (DIMENSIONS, METHODS as STATIC_METHODS, STATIC_DIR, DYNAMIC_DIR,
                    PREDICTION_DIR, cluster_bootstrap, csv_write, evaluate_predictions,
                    format_float, jsonl_read, latest_rows_by_pair, load_split_pairs,
                    predictions_from_jsonl, current_profile, gold_add, load_dynamic_split,
                    safe_div, span_exists, load_approved_gold_rows)

PREDICTION_FILES = {
    "keyword": PREDICTION_DIR / "keyword_test.jsonl",
    "embedding": PREDICTION_DIR / "embedding_test.jsonl",
    "llm_only": PREDICTION_DIR / "llm_only_test.jsonl",
    "llm_vocab": PREDICTION_DIR / "llm_vocab_test.jsonl",
    "igdst": PREDICTION_DIR / "igdst_test.jsonl",
}


def evaluate_static(n_bootstrap: int = 2000) -> dict:
    pairs = load_split_pairs("test")
    expected_pair_ids = {pair["pair_id"] for pair in pairs}
    results = {}
    for method, path in PREDICTION_FILES.items():
        if not path.exists():
            results[method] = {"status": "missing_predictions", "path": str(path)}
            continue
        extra_pair_ids = set()
        latest_rows = latest_rows_by_pair(path, expected_pair_ids)
        for row in jsonl_read(path):
            pair_id = row.get("pair_id")
            if pair_id and pair_id not in expected_pair_ids:
                extra_pair_ids.add(pair_id)
        failed_rows = [
            row for row in latest_rows.values()
            if row.get("status") != "success"
        ]
        success_pair_ids = {
            pair_id
            for pair_id, row in latest_rows.items()
            if row.get("status") == "success"
        }
        if failed_rows or success_pair_ids != expected_pair_ids or set(latest_rows) != expected_pair_ids:
            results[method] = {
                "status": "incomplete_predictions",
                "path": str(path),
                "success_pairs": len(success_pair_ids),
                "required_pairs": len(pairs),
                "failed_records": len(failed_rows),
                "duplicate_pair_ids": [],
                "extra_pair_ids": sorted(extra_pair_ids),
            }
            continue
        preds = predictions_from_jsonl(path)
        metrics = evaluate_predictions(pairs, preds)
        ci = cluster_bootstrap(pairs, preds, n_bootstrap=n_bootstrap)
        results[method] = {
            "status": "complete",
            "path": str(path),
            "metrics": metrics,
            "ci": ci,
            "failed_records": 0,
        }
    return results


def main_accuracy_rows(results: dict) -> list[dict]:
    rows = []
    for method, label in STATIC_METHODS.items():
        item = results.get(method, {})
        if item.get("status") != "complete":
            rows.append(
                {
                    "Method": label,
                    "Status": item.get("status", "missing_predictions"),
                    "Hazard F1": "",
                    "Stage F1": "",
                    "Accident F1": "",
                    "Overall Macro-F1": "",
                    "Jaccard": "",
                }
            )
            continue
        m = item["metrics"]
        ci = item["ci"]
        by = m["by_dimension"]
        jaccard = sum(by[dim]["jaccard"] for dim in DIMENSIONS) / len(DIMENSIONS)
        rows.append(
            {
                "Method": label,
                "Status": "complete",
                "Hazard F1": ci_string(by["hazard"]["f1"], ci["hazard_f1"]),
                "Stage F1": ci_string(by["operation_stage"]["f1"], ci["operation_stage_f1"]),
                "Accident F1": ci_string(by["accident_type"]["f1"], ci["accident_type_f1"]),
                "Overall Macro-F1": ci_string(m["overall_macro_f1"], ci["overall_macro_f1"]),
                "Jaccard": format_float(jaccard),
            }
        )
    return rows


def ci_string(estimate: float, ci: tuple[float, float]) -> str:
    return f"{format_float(estimate)} [{format_float(ci[0])}, {format_float(ci[1])}]"


def static_bootstrap_rows(results: dict) -> list[dict]:
    rows = []
    for method, label in STATIC_METHODS.items():
        item = results.get(method, {})
        if item.get("status") != "complete":
            continue
        metrics = item["metrics"]
        ci = item["ci"]
        rows.append({"Method": label, "Metric": "Overall Macro-F1", "Estimate": metrics["overall_macro_f1"], "CI_low": ci["overall_macro_f1"][0], "CI_high": ci["overall_macro_f1"][1]})
        for dim, metric_name in [("hazard", "Hazard F1"), ("operation_stage", "Stage F1"), ("accident_type", "Accident F1")]:
            rows.append({"Method": label, "Metric": metric_name, "Estimate": metrics["by_dimension"][dim]["f1"], "CI_low": ci[f"{dim}_f1"][0], "CI_high": ci[f"{dim}_f1"][1]})
    return rows


def per_label_rows(results: dict) -> list[dict]:
    rows = []
    for method, label in STATIC_METHODS.items():
        item = results.get(method, {})
        if item.get("status") != "complete":
            continue
        for row in item["metrics"]["per_label"]:
            rows.append({"method": label, **row})
    return rows


def sensitivity_rows(results: dict) -> list[dict]:
    rows = []
    for method, label in STATIC_METHODS.items():
        item = results.get(method, {})
        if item.get("status") != "complete":
            continue
        for dim in DIMENSIONS:
            m = item["metrics"]["by_dimension"][dim]
            rows.append(
                {
                    "method": label,
                    "dimension": dim,
                    "positive_gold_count": m["positive_gold_count"],
                    "positive_recall": m["positive_recall"],
                    "positive_jaccard": m["positive_jaccard"],
                    "positive_exact_match": m["positive_exact_match"],
                    "exact_empty_agreement_rate": m["exact_empty_agreement_rate"],
                    "empty_gold_count": m["empty_gold_count"],
                }
            )
    return rows


def paired_comparison_rows(results: dict, n_bootstrap: int = 2000) -> list[dict]:
    pairs = load_split_pairs("test")
    if results.get("igdst", {}).get("status") != "complete":
        return []
    igdst_preds = predictions_from_jsonl(PREDICTION_FILES["igdst"])
    rows = []
    by_acc = defaultdict(list)
    for pair in pairs:
        cluster_id = pair.get("accident_group_id") or pair["accident_id"]
        by_acc[cluster_id].append(pair)
    accident_ids = sorted(by_acc)
    rng = random.Random(42)
    for method in ["keyword", "embedding", "llm_only", "llm_vocab"]:
        if results.get(method, {}).get("status") != "complete":
            continue
        other_preds = predictions_from_jsonl(PREDICTION_FILES[method])
        diffs = []
        for _ in range(n_bootstrap):
            sample = []
            for _ in accident_ids:
                sample.extend(by_acc[rng.choice(accident_ids)])
            ig = evaluate_predictions(sample, igdst_preds)["overall_macro_f1"]
            ot = evaluate_predictions(sample, other_preds)["overall_macro_f1"]
            diffs.append(ig - ot)
        diffs_sorted = sorted(diffs)
        rows.append(
            {
                "comparison": f"IGDST - {STATIC_METHODS[method]}",
                "mean_difference": sum(diffs) / len(diffs),
                "CI_low": diffs_sorted[int(0.025 * (len(diffs_sorted) - 1))],
                "CI_high": diffs_sorted[int(0.975 * (len(diffs_sorted) - 1))],
                "interpretation": "bootstrap-supported improvement" if diffs_sorted[int(0.025 * (len(diffs_sorted) - 1))] > 0 else "",
            }
        )
    return rows

DYNAMIC_METHODS = {
    "d1": {
        "label": "Rule",
        "path": DYNAMIC_DIR / "rule_dynamic_test.jsonl",
    },
    "d2": {
        "label": "LLM-NewAccident",
        "path": DYNAMIC_DIR / "llm_new_accident_test.jsonl",
    },
    "d3": {
        "label": "LLM+CurrentProfile",
        "path": DYNAMIC_DIR / "llm_profile_test.jsonl",
    },
    "d4": {
        "label": "IGDST Dynamic",
        "path": DYNAMIC_DIR / "igdst_dynamic_test.jsonl",
    },
}


IGDST_DYNAMIC_VERSION = "phase3_dynamic_v2"


def prediction_path(method: str, split: str) -> Path:
    names = {
        "d1": "rule_dynamic",
        "d2": "llm_new_accident",
        "d3": "llm_profile",
        "d4": "igdst_dynamic",
    }
    return DYNAMIC_DIR / f"{names[method]}_{split}.jsonl"


MAIN_FIELDS = [
    "Method",
    "Status",
    "Update Precision",
    "Update Recall",
    "Update F1",
    "Dynamic Macro-F1",
    "False Update Rate",
    "Missed Update Rate",
    "Novel Tag Precision",
]




def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalize_add(value: Any) -> dict[str, set[str]]:
    if not isinstance(value, dict):
        raise ValueError("predicted_add must be an object.")
    result = {}
    for dimension in DIMENSIONS:
        items = value.get(dimension)
        if not isinstance(items, list):
            raise ValueError(f"predicted_add.{dimension} must be an array.")
        result[dimension] = {
            str(item).strip() for item in items if str(item).strip()
        }
    return result


def prediction_completeness(
    cases: list[dict[str, str]],
    path: Path,
    allow_extra: bool = False,
    required_prompt_version: str | None = None,
) -> dict[str, Any]:
    expected_ids = {row["dynamic_case_id"] for row in cases}
    if not path.is_file():
        return {
            "status": "incomplete_predictions",
            "path": str(path),
            "predictions": {},
            "rows": [],
            "failed": 0,
            "issues": ["prediction file is missing"],
        }
    all_rows = jsonl_read(path)
    rows = (
        [row for row in all_rows if row.get("dynamic_case_id") in expected_ids]
        if allow_extra
        else all_rows
    )
    counts = Counter(row.get("dynamic_case_id", "") for row in rows)
    failed = sum(row.get("status") != "success" for row in rows)
    success_ids = {
        row.get("dynamic_case_id")
        for row in rows
        if row.get("status") == "success"
    }
    issues = []
    if set(counts) != expected_ids:
        issues.append(
            f"case set mismatch: missing={len(expected_ids - set(counts))}, "
            f"extra={len(set(counts) - expected_ids)}"
        )
    duplicates = sorted(case_id for case_id, count in counts.items() if count != 1)
    if duplicates:
        issues.append(f"duplicate case ids={len(duplicates)}")
    if success_ids != expected_ids:
        issues.append(
            f"success set mismatch: missing={len(expected_ids - success_ids)}, "
            f"extra={len(success_ids - expected_ids)}"
        )
    if failed:
        issues.append(f"failed rows={failed}")
    if required_prompt_version:
        stale = sum(
            row.get("prompt_version") != required_prompt_version for row in rows
        )
        if stale:
            issues.append(
                f"stale prompt_version rows={stale}; "
                f"required={required_prompt_version}"
            )

    predictions = {}
    if not issues:
        try:
            for row in rows:
                action = str(row.get("predicted_action", "")).strip().upper()
                if action not in {"ADD", "NO_CHANGE"}:
                    raise ValueError(
                        f"{row.get('dynamic_case_id')} has invalid predicted_action."
                    )
                additions = normalize_add(row.get("predicted_add"))
                expected_action = "ADD" if any(additions.values()) else "NO_CHANGE"
                if action != expected_action:
                    raise ValueError(
                        f"{row.get('dynamic_case_id')} action/add inconsistency."
                    )
                predictions[row["dynamic_case_id"]] = {
                    "action": action,
                    "add": additions,
                }
        except ValueError as exc:
            issues.append(str(exc))
            predictions = {}
    return {
        "status": "complete" if not issues else "incomplete_predictions",
        "path": str(path),
        "predictions": predictions,
        "rows": rows,
        "failed": failed,
        "issues": issues,
    }


def evaluate_metrics(
    cases: list[dict[str, str]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    tag_counts = {
        dimension: {"tp": 0, "fp": 0, "fn": 0} for dimension in DIMENSIONS
    }
    all_tag_tp = all_tag_pred = 0
    for row in cases:
        case_id = row["dynamic_case_id"]
        gold_action = row["gold_action"]
        pred_action = predictions[case_id]["action"]
        if gold_action == "ADD" and pred_action == "ADD":
            tp += 1
        elif gold_action == "NO_CHANGE" and pred_action == "ADD":
            fp += 1
        elif gold_action == "ADD" and pred_action == "NO_CHANGE":
            fn += 1
        else:
            tn += 1

        expected = gold_add(row)
        proposed = predictions[case_id]["add"]
        for dimension in DIMENSIONS:
            tag_counts[dimension]["tp"] += len(expected[dimension] & proposed[dimension])
            tag_counts[dimension]["fp"] += len(proposed[dimension] - expected[dimension])
            tag_counts[dimension]["fn"] += len(expected[dimension] - proposed[dimension])
            all_tag_tp += len(expected[dimension] & proposed[dimension])
            all_tag_pred += len(proposed[dimension])

    update_precision = safe_div(tp, tp + fp)
    update_recall = safe_div(tp, tp + fn)
    update_f1 = f1_score(update_precision, update_recall)
    specificity = safe_div(tn, tn + fp)
    dimension_metrics = {}
    for dimension in DIMENSIONS:
        counts = tag_counts[dimension]
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        dimension_metrics[dimension] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
        }
    return {
        "n": len(cases),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "update_precision": update_precision,
        "update_recall": update_recall,
        "update_f1": update_f1,
        "accuracy": safe_div(tp + tn, len(cases)),
        "balanced_accuracy": (update_recall + specificity) / 2,
        "false_update_rate": safe_div(fp, fp + tn),
        "missed_update_rate": safe_div(fn, tp + fn),
        "novel_tag_precision": safe_div(all_tag_tp, all_tag_pred),
        "dynamic_macro_f1": sum(
            dimension_metrics[dimension]["f1"] for dimension in DIMENSIONS
        )
        / len(DIMENSIONS),
        "by_dimension": dimension_metrics,
    }


def evaluate_dynamic(
    cases: list[dict[str, str]], split: str = "test"
) -> dict[str, dict[str, Any]]:
    results = {}
    for key, config in DYNAMIC_METHODS.items():
        completeness = prediction_completeness(
            cases,
            prediction_path(key, split),
            allow_extra=split == "dev",
            required_prompt_version=(
                IGDST_DYNAMIC_VERSION if key == "d4" else None
            ),
        )
        result = {**completeness, "label": config["label"]}
        if completeness["status"] == "complete":
            result["metrics"] = evaluate_metrics(cases, completeness["predictions"])
        results[key] = result
    return results


def dynamic_main_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in DYNAMIC_METHODS:
        result = results[key]
        if result["status"] != "complete":
            rows.append(
                {
                    "Method": result["label"],
                    "Status": "incomplete_predictions",
                    **{field: "" for field in MAIN_FIELDS[2:]},
                }
            )
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "Method": result["label"],
                "Status": "complete",
                "Update Precision": metrics["update_precision"],
                "Update Recall": metrics["update_recall"],
                "Update F1": metrics["update_f1"],
                "Dynamic Macro-F1": metrics["dynamic_macro_f1"],
                "False Update Rate": metrics["false_update_rate"],
                "Missed Update Rate": metrics["missed_update_rate"],
                "Novel Tag Precision": metrics["novel_tag_precision"],
            }
        )
    return rows


def dimension_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in DYNAMIC_METHODS:
        result = results[key]
        if result["status"] != "complete":
            continue
        by = result["metrics"]["by_dimension"]
        rows.append(
            {
                "Method": result["label"],
                "Hazard Add F1": by["hazard"]["f1"],
                "Stage Add F1": by["operation_stage"]["f1"],
                "Accident Add F1": by["accident_type"]["f1"],
            }
        )
    return rows


def dynamic_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(quantile * (len(ordered) - 1))
    return ordered[index]


def dynamic_bootstrap_rows(
    cases: list[dict[str, str]],
    results: dict[str, dict[str, Any]],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cases:
        cluster = row.get("accident_group_id") or row["new_accident_id"]
        by_cluster[cluster].append(row)
    cluster_ids = sorted(by_cluster)
    metric_names = {
        "Update F1": "update_f1",
        "Dynamic Macro-F1": "dynamic_macro_f1",
        "False Update Rate": "false_update_rate",
        "Missed Update Rate": "missed_update_rate",
    }
    output = []
    for key in DYNAMIC_METHODS:
        result = results[key]
        if result["status"] != "complete":
            continue
        rng = random.Random(42)
        samples = {metric: [] for metric in metric_names.values()}
        for _ in range(n_bootstrap):
            sampled_cases = []
            for _ in cluster_ids:
                sampled_cases.extend(by_cluster[rng.choice(cluster_ids)])
            metrics = evaluate_metrics(sampled_cases, result["predictions"])
            for metric in samples:
                samples[metric].append(metrics[metric])
        for label, metric in metric_names.items():
            output.append(
                {
                    "Method": result["label"],
                    "Metric": label,
                    "Estimate": result["metrics"][metric],
                    "CI_low": dynamic_percentile(samples[metric], 0.025),
                    "CI_high": dynamic_percentile(samples[metric], 0.975),
                    "n_bootstrap": n_bootstrap,
                    "seed": 42,
                    "cluster_unit": "accident_group_id (fallback new_accident_id)",
                }
            )
    return output


def ablation_predictions_without_verifier(
    cases: list[dict[str, str]],
    d4_rows: list[dict[str, Any]],
    vocabulary: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    source_by_id = {row["dynamic_case_id"]: row for row in cases}
    predictions = {}
    for prediction in d4_rows:
        case_id = prediction["dynamic_case_id"]
        source = source_by_id[case_id]
        profile = current_profile(source)
        additions = {dimension: set() for dimension in DIMENSIONS}
        for candidate in prediction.get("generator_candidates", []):
            dimension = candidate.get("dimension", "")
            tag = str(candidate.get("tag", "")).strip()
            if dimension not in DIMENSIONS:
                continue
            if tag not in vocabulary[dimension] or tag in profile[dimension]:
                continue
            if not span_exists(
                candidate.get("evidence_span", ""), source["new_accident_description"]
            ):
                continue
            additions[dimension].add(tag)
        action = "ADD" if any(additions.values()) else "NO_CHANGE"
        predictions[case_id] = {"action": action, "add": additions}
    return predictions


def ablation_predictions_without_profile_filter(
    cases: list[dict[str, str]],
    d4_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Remove only the deterministic Current Profile set difference."""
    expected_ids = {row["dynamic_case_id"] for row in cases}
    predictions = {}
    for prediction in d4_rows:
        case_id = prediction.get("dynamic_case_id", "")
        if case_id not in expected_ids:
            continue
        additions = {dimension: set() for dimension in DIMENSIONS}
        for verdict in prediction.get("verifier_results", []):
            dimension = verdict.get("dimension", "")
            tag = str(verdict.get("tag", "")).strip()
            if (
                dimension in DIMENSIONS
                and tag
                and verdict.get("verdict") == "SUPPORTED"
            ):
                additions[dimension].add(tag)
        action = "ADD" if any(additions.values()) else "NO_CHANGE"
        predictions[case_id] = {"action": action, "add": additions}
    return predictions


def ablation_rows(
    cases: list[dict[str, str]], results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    variants: list[tuple[str, dict[str, dict[str, Any]]]] = []
    if results["d4"]["status"] == "complete":
        variants.append(("Full IGDST Dynamic", results["d4"]["predictions"]))
    if results["d4"]["status"] == "complete":
        _, _, vocabulary, _, _ = load_approved_gold_rows()
        variants.append(
            (
                "w/o Deterministic Profile Filter",
                ablation_predictions_without_profile_filter(
                    cases, results["d4"]["rows"]
                ),
            )
        )
        variants.append(
            (
                "w/o Verifier",
                ablation_predictions_without_verifier(
                    cases, results["d4"]["rows"], vocabulary
                ),
            )
        )
    for label, predictions in variants:
        metrics = evaluate_metrics(cases, predictions)
        rows.append(
            {
                "Model": label,
                "Update F1": metrics["update_f1"],
                "Dynamic Macro-F1": metrics["dynamic_macro_f1"],
                "False Update Rate": metrics["false_update_rate"],
                "Missed Update Rate": metrics["missed_update_rate"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete prediction files without API calls.")
    parser.add_argument("task", choices=["static", "dynamic"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    if args.task == "static":
        results = evaluate_static(args.n_bootstrap)
        rows = main_accuracy_rows(results)
        destination = STATIC_DIR
        ci = static_bootstrap_rows(results)
        for name, extra in [("per_label_metrics", per_label_rows(results)),
                            ("sensitivity_analysis", sensitivity_rows(results)),
                            ("paired_bootstrap_comparison", paired_comparison_rows(results, args.n_bootstrap))]:
            if extra:
                csv_write(destination / f"{name}.csv", extra, list(extra[0]))
    else:
        cases = load_dynamic_split("test")
        results = evaluate_dynamic(cases)
        rows = dynamic_main_rows(results)
        destination = DYNAMIC_DIR
        ci = dynamic_bootstrap_rows(cases, results, args.n_bootstrap)
        for name, extra in [("table_dynamic_dimensions", dimension_rows(results)),
                            ("table_dynamic_ablation", ablation_rows(cases, results))]:
            if extra:
                csv_write(destination / f"{name}.csv", extra, list(extra[0]))
    csv_write(destination / "table_main.csv", rows, list(rows[0]))
    if ci:
        csv_write(destination / "bootstrap_ci.csv", ci, list(ci[0]))
    for method, result in results.items():
        print(method, result["status"])
    return 0 if any(result["status"] == "complete" for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
