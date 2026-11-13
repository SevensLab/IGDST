"""Prepare the static and temporal benchmarks from final annotations."""
from __future__ import annotations
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from common import (DIMENSIONS, STATIC_DIR, DYNAMIC_DIR, build_benchmark_pair, csv_write,
                    description_hash, load_gold_data, parse_multilabel, enrich_gold_rows,
                    parse_date, tag_set, format_tag_set)

PAIR_FIELDS = [
    "row_id",
    "pair_id",
    "accident_id",
    "date",
    "resource_id",
    "resource_name",
    "resource_type",
    "description",
    "gold_include_resource",
    "gold_hazard",
    "gold_operation_stage",
    "gold_accident_type",
    "accident_group_id",
    "description_hash",
]


def build_static_split(workbook: Path | None = None, seed: int = 42, dev_fraction: float = 0.2) -> tuple[list[dict], list[dict], dict]:
    annotations, contexts, vocab_by_dimension, _, _ = load_gold_data(workbook)
    include_rows = [row for row in annotations if row.get("gold_include_resource", "").strip().lower() == "yes"]

    hash_to_group_id: dict[str, str] = {}
    group_to_accidents: dict[str, set[str]] = defaultdict(set)
    for accident_id, ctx in contexts.items():
        h = description_hash(ctx.get("description", ""))
        group_id = hash_to_group_id.setdefault(h, f"G{len(hash_to_group_id)+1:04d}")
        group_to_accidents[group_id].add(accident_id)

    pairs = []
    for row in include_rows:
        accident_id = row.get("accident_id", "")
        ctx = contexts[accident_id]
        h = description_hash(ctx.get("description", ""))
        group_id = hash_to_group_id[h]
        pair = build_benchmark_pair(row, ctx)
        pair["accident_group_id"] = group_id
        pair["description_hash"] = h
        pairs.append(pair)

    groups = sorted({p["accident_group_id"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    dev_n = max(1, round(len(groups) * dev_fraction))
    dev_groups = set(groups[:dev_n])
    dev_pairs = [p for p in pairs if p["accident_group_id"] in dev_groups]
    test_pairs = [p for p in pairs if p["accident_group_id"] not in dev_groups]

    stats = split_stats(dev_pairs, test_pairs, vocab_by_dimension)
    stats["seed"] = seed
    stats["dev_fraction"] = dev_fraction
    stats["total_groups"] = len(groups)
    stats["dev_groups"] = len(dev_groups)
    stats["test_groups"] = len(groups) - len(dev_groups)
    stats["duplicate_group_count"] = sum(1 for ids in group_to_accidents.values() if len(ids) > 1)
    return dev_pairs, test_pairs, stats


def split_stats(dev_pairs: list[dict], test_pairs: list[dict], vocab_by_dimension: dict[str, set[str]]) -> dict:
    out = {}
    for split_name, rows in [("dev", dev_pairs), ("test", test_pairs)]:
        out[split_name] = {
            "pairs": len(rows),
            "accidents": len({r["accident_id"] for r in rows}),
            "groups": len({r["accident_group_id"] for r in rows}),
            "resource_type_distribution": dict(Counter(r["resource_type"] for r in rows)),
            "tag_distribution": {},
        }
        for dim in DIMENSIONS:
            c = Counter()
            empty = 0
            for row in rows:
                tags = parse_multilabel(row[f"gold_{dim}"])
                if not tags:
                    empty += 1
                c.update(tags)
            out[split_name]["tag_distribution"][dim] = {"empty": empty, "tags": dict(c)}
    out["rare_label_coverage_limitations"] = []
    for dim in DIMENSIONS:
        for tag in sorted(vocab_by_dimension[dim]):
            dev_support = out["dev"]["tag_distribution"][dim]["tags"].get(tag, 0)
            test_support = out["test"]["tag_distribution"][dim]["tags"].get(tag, 0)
            total = dev_support + test_support
            if total <= 2 and (dev_support == 0 or test_support == 0):
                out["rare_label_coverage_limitations"].append(
                    {"dimension": dim, "tag": tag, "dev_support": dev_support, "test_support": test_support, "total_support": total}
                )
    return out

PROFILE_SOURCE = "past_human_gold_oracle"


ELIGIBILITY_FIELDS = [
    "resource_id",
    "resource_name",
    "resource_type",
    "n_accidents",
    "n_unique_accident_groups",
    "first_date",
    "last_date",
    "eligible_ge3",
    "eligible_ge5",
]


CASE_FIELDS = [
    "dynamic_case_id",
    "resource_id",
    "resource_name",
    "resource_type",
    "profile_source",
    "current_time",
    "new_accident_id",
    "new_accident_date",
    "new_accident_description",
    "current_hazard_tags",
    "current_operation_stage_tags",
    "current_accident_type_tags",
    "gold_new_hazard_tags",
    "gold_new_operation_stage_tags",
    "gold_new_accident_type_tags",
    "gold_add_hazard_tags",
    "gold_add_operation_stage_tags",
    "gold_add_accident_type_tags",
    "gold_action",
    "history_accident_count",
    "history_accident_ids",
    "profile_total_tags_before",
    "accident_group_id",
    "description_hash",
    "duplicate_group_id",
    "excluded_duplicate_dynamic_case",
    "included_in_benchmark",
    "case_status",
    "source_pair_id",
    "source_gold_row_id",
]


def validate_cases(
    eligibility: list[dict[str, str]],
    cases: list[dict[str, str]],
    accident_dates: dict[str, object],
) -> None:
    for row in eligibility:
        unique_groups = int(row["n_unique_accident_groups"])
        if row["eligible_ge3"] == "true" and unique_groups < 3:
            raise ValueError("eligible_ge3 invariant violated.")
        if row["eligible_ge5"] == "true" and unique_groups < 5:
            raise ValueError("eligible_ge5 invariant violated.")

    retained_groups = Counter(
        (row["resource_id"], row["accident_group_id"])
        for row in cases
        if row["excluded_duplicate_dynamic_case"] != "true"
    )
    if any(count > 1 for count in retained_groups.values()):
        raise ValueError("Duplicate accident group retained more than once for a resource.")

    current_by_resource_date: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for row in cases:
        if row["included_in_benchmark"] != "true":
            continue
        current_date = parse_date(row["new_accident_date"])
        history_ids = [value for value in row["history_accident_ids"].split(";") if value]
        if int(row["history_accident_count"]) != len(history_ids):
            raise ValueError("history_accident_count invariant violated.")
        for accident_id in history_ids:
            history_date = accident_dates.get(accident_id)
            if history_date is None or history_date >= current_date:
                raise ValueError("Temporal leakage detected.")
        signature = tuple(row[f"current_{dimension}_tags"] for dimension in DIMENSIONS)
        current_by_resource_date[(row["resource_id"], row["new_accident_date"])].add(signature)
        for dimension in DIMENSIONS:
            current = tag_set(row, dimension, prefix="current_")
            gold_new = tag_set(row, dimension, prefix="gold_new_")
            gold_add = tag_set(row, dimension, prefix="gold_add_")
            if gold_add != gold_new - current:
                raise ValueError("Gold ADD invariant violated.")
        expected_action = "ADD" if any(
            tag_set(row, dimension, prefix="gold_add_") for dimension in DIMENSIONS
        ) else "NO_CHANGE"
        if row["gold_action"] != expected_action:
            raise ValueError("Gold action invariant violated.")
    if any(len(signatures) != 1 for signatures in current_by_resource_date.values()):
        raise ValueError("Same-date current profile invariant violated.")


def build_dynamic_cases() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    rows, _, _, _, _, _ = enrich_gold_rows()
    by_resource: dict[str, list[dict[str, str]]] = defaultdict(list)
    accident_dates: dict[str, object] = {}
    for row in rows:
        by_resource[row["resource_id"]].append(row)
        parsed = parse_date(row["date"])
        existing = accident_dates.get(row["accident_id"])
        if existing is not None and existing != parsed:
            raise ValueError(f"Conflicting dates for accident {row['accident_id']}.")
        accident_dates[row["accident_id"]] = parsed

    eligibility = []
    eligible_resources: set[str] = set()
    for resource_id, resource_rows in sorted(by_resource.items()):
        raw_accidents = {row["accident_id"] for row in resource_rows}
        unique_groups = {row["accident_group_id"] for row in resource_rows}
        dates = [parse_date(row["date"]) for row in resource_rows]
        group_count = len(unique_groups)
        item = {
            "resource_id": resource_id,
            "resource_name": resource_rows[0].get("resource_name", ""),
            "resource_type": resource_rows[0].get("resource_type", ""),
            "n_accidents": len(raw_accidents),
            "n_unique_accident_groups": group_count,
            "first_date": min(dates).isoformat(),
            "last_date": max(dates).isoformat(),
            "eligible_ge3": "true" if group_count >= 3 else "false",
            "eligible_ge5": "true" if group_count >= 5 else "false",
        }
        eligibility.append(item)
        if group_count >= 3:
            eligible_resources.add(resource_id)

    cases = []
    case_counter = 0
    action_counts = Counter()
    status_counts = Counter()

    for resource_id in sorted(eligible_resources):
        resource_rows = sorted(
            by_resource[resource_id],
            key=lambda row: (parse_date(row["date"]), row["accident_id"], row["pair_id"]),
        )
        rows_by_date: dict[object, list[dict[str, str]]] = defaultdict(list)
        for row in resource_rows:
            rows_by_date[parse_date(row["date"])].append(row)

        profile = {dimension: set() for dimension in DIMENSIONS}
        history_ids: list[str] = []
        seen_groups: set[str] = set()
        resource_name = resource_rows[0].get("resource_name", "")
        resource_type = resource_rows[0].get("resource_type", "")

        for batch_date in sorted(rows_by_date):
            batch_rows = sorted(
                rows_by_date[batch_date],
                key=lambda row: (row["accident_id"], row["pair_id"]),
            )
            retained_rows = []
            duplicate_by_pair: dict[str, bool] = {}
            for row in batch_rows:
                group_id = row["accident_group_id"]
                is_duplicate = group_id in seen_groups
                duplicate_by_pair[row["pair_id"]] = is_duplicate
                if not is_duplicate:
                    seen_groups.add(group_id)
                    retained_rows.append(row)

            current_profile = {
                dimension: set(profile[dimension]) for dimension in DIMENSIONS
            }
            history_snapshot = list(history_ids)
            initializing_batch = not history_snapshot

            for row in batch_rows:
                is_duplicate = duplicate_by_pair[row["pair_id"]]
                new_tags = {
                    dimension: tag_set(row, dimension) for dimension in DIMENSIONS
                }
                additions = {
                    dimension: new_tags[dimension] - current_profile[dimension]
                    for dimension in DIMENSIONS
                }
                action = "ADD" if any(additions.values()) else "NO_CHANGE"

                if is_duplicate:
                    status = "excluded_duplicate_dynamic_case"
                    included = "false"
                    dynamic_case_id = ""
                    action_value = ""
                elif initializing_batch:
                    status = "initialization"
                    included = "false"
                    dynamic_case_id = ""
                    action_value = ""
                else:
                    status = "dynamic_case"
                    included = "true"
                    case_counter += 1
                    dynamic_case_id = f"DYN_{case_counter:06d}"
                    action_value = action
                    action_counts[action] += 1

                case = {
                    "dynamic_case_id": dynamic_case_id,
                    "resource_id": resource_id,
                    "resource_name": resource_name,
                    "resource_type": resource_type,
                    "profile_source": PROFILE_SOURCE,
                    "current_time": batch_date.isoformat(),
                    "new_accident_id": row["accident_id"],
                    "new_accident_date": batch_date.isoformat(),
                    "new_accident_description": row["description"],
                    "current_hazard_tags": format_tag_set(current_profile["hazard"]),
                    "current_operation_stage_tags": format_tag_set(current_profile["operation_stage"]),
                    "current_accident_type_tags": format_tag_set(current_profile["accident_type"]),
                    "gold_new_hazard_tags": format_tag_set(new_tags["hazard"]),
                    "gold_new_operation_stage_tags": format_tag_set(new_tags["operation_stage"]),
                    "gold_new_accident_type_tags": format_tag_set(new_tags["accident_type"]),
                    "gold_add_hazard_tags": format_tag_set(additions["hazard"]),
                    "gold_add_operation_stage_tags": format_tag_set(additions["operation_stage"]),
                    "gold_add_accident_type_tags": format_tag_set(additions["accident_type"]),
                    "gold_action": action_value,
                    "history_accident_count": len(history_snapshot),
                    "history_accident_ids": ";".join(history_snapshot),
                    "profile_total_tags_before": sum(
                        len(current_profile[dimension]) for dimension in DIMENSIONS
                    ),
                    "accident_group_id": row["accident_group_id"],
                    "description_hash": row["description_hash"],
                    "duplicate_group_id": row["duplicate_group_id"],
                    "excluded_duplicate_dynamic_case": "true" if is_duplicate else "false",
                    "included_in_benchmark": included,
                    "case_status": status,
                    "source_pair_id": row["pair_id"],
                    "source_gold_row_id": row.get("row_id", ""),
                }
                cases.append(case)
                status_counts[status] += 1

            for row in retained_rows:
                for dimension in DIMENSIONS:
                    profile[dimension].update(tag_set(row, dimension))
                history_ids.append(row["accident_id"])

    validate_cases(eligibility, cases, accident_dates)
    stats = {
        "profile_source": PROFILE_SOURCE,
        "approved_gold_rows": len(rows),
        "eligible_resource_total": len(eligibility),
        "eligible_ge3_resources": sum(
            row["eligible_ge3"] == "true" for row in eligibility
        ),
        "eligible_ge5_resources": sum(
            row["eligible_ge5"] == "true" for row in eligibility
        ),
        "dynamic_cases": sum(
            row["included_in_benchmark"] == "true" for row in cases
        ),
        "initialization_rows": status_counts["initialization"],
        "excluded_duplicate_rows": status_counts["excluded_duplicate_dynamic_case"],
        "gold_add_cases": action_counts["ADD"],
        "gold_no_change_cases": action_counts["NO_CHANGE"],
        "action_distribution": dict(action_counts),
        "resource_type_distribution": dict(
            Counter(
                row["resource_type"]
                for row in eligibility
                if row["eligible_ge3"] == "true"
            )
        ),
    }
    return eligibility, cases, stats

SPLIT_FIELDS = [
    "dynamic_case_id",
    "resource_id",
    "resource_name",
    "resource_type",
    "profile_source",
    "current_time",
    "new_accident_id",
    "new_accident_date",
    "new_accident_description",
    "current_hazard_tags",
    "current_operation_stage_tags",
    "current_accident_type_tags",
    "gold_new_hazard_tags",
    "gold_new_operation_stage_tags",
    "gold_new_accident_type_tags",
    "gold_add_hazard_tags",
    "gold_add_operation_stage_tags",
    "gold_add_accident_type_tags",
    "gold_action",
    "history_accident_count",
    "history_accident_ids",
    "profile_total_tags_before",
    "accident_group_id",
    "description_hash",
    "duplicate_group_id",
    "source_pair_id",
    "source_gold_row_id",
    "split",
]


def distribution(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(field, "") for row in rows).items()))


def choose_cutoff(
    cases: list[dict[str, str]], target_fraction: float
) -> object | None:
    by_date: dict[object, list[dict[str, str]]] = defaultdict(list)
    for row in cases:
        by_date[parse_date(row["new_accident_date"])].append(row)
    dates = sorted(by_date)
    candidates = []
    for cutoff in dates[:-1]:
        dev = [row for row in cases if parse_date(row["new_accident_date"]) <= cutoff]
        test = [row for row in cases if parse_date(row["new_accident_date"]) > cutoff]
        if not dev or not test:
            continue
        accident_overlap = {
            row["new_accident_id"] for row in dev
        } & {row["new_accident_id"] for row in test}
        group_overlap = {
            row["accident_group_id"] for row in dev
        } & {row["accident_group_id"] for row in test}
        if accident_overlap or group_overlap:
            continue
        actual_fraction = len(dev) / len(cases)
        candidates.append((abs(actual_fraction - target_fraction), cutoff, cutoff))
    return min(candidates)[2] if candidates else None


def split_status(stats: dict) -> str:
    if not stats["cutoff_date"] or stats["strict_chronological_boundary"] != "PASS":
        return "FAIL"
    if stats["split_date_overlap"]:
        return "FAIL"
    required = [
        stats["dev_cases"] > 0,
        stats["test_cases"] > 0,
        not stats["split_accident_overlap"],
        not stats["split_group_overlap"],
        not stats["split_date_overlap"],
        stats["test_action_distribution"].get("ADD", 0) > 0,
        stats["test_action_distribution"].get("NO_CHANGE", 0) > 0,
    ]
    return "PASS" if all(required) else "CHECK_REQUIRED"


def build_temporal_split(all_rows) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    cases = [
        row
        for row in all_rows
        if row.get("included_in_benchmark") == "true"
        and row.get("excluded_duplicate_dynamic_case") != "true"
    ]
    cases.sort(
        key=lambda row: (parse_date(row["new_accident_date"]), row["dynamic_case_id"])
    )
    target_fraction = 0.30 if len(cases) >= 200 else 0.40
    cutoff = choose_cutoff(cases, target_fraction) if cases else None

    dev = []
    test = []
    if cutoff is not None:
        for row in cases:
            split = "dev" if parse_date(row["new_accident_date"]) <= cutoff else "test"
            output = {field: row.get(field, "") for field in SPLIT_FIELDS}
            output["split"] = split
            (dev if split == "dev" else test).append(output)

    dev_dates = {row["new_accident_date"] for row in dev}
    test_dates = {row["new_accident_date"] for row in test}
    dev_accidents = {row["new_accident_id"] for row in dev}
    test_accidents = {row["new_accident_id"] for row in test}
    dev_groups = {row["accident_group_id"] for row in dev}
    test_groups = {row["accident_group_id"] for row in test}
    dev_last = max(dev_dates) if dev_dates else ""
    test_first = min(test_dates) if test_dates else ""
    chronological = bool(dev and test and parse_date(dev_last) < parse_date(test_first))
    stats = {
        "total_dynamic_cases": len(cases),
        "total_accident_groups": len(
            {row["accident_group_id"] for row in cases}
        ),
        "target_dev_fraction": target_fraction,
        "actual_dev_fraction": len(dev) / len(cases) if cases else 0.0,
        "cutoff_date": cutoff.isoformat() if cutoff is not None else "",
        "dev_groups": len(dev_groups),
        "test_groups": len(test_groups),
        "dev_cases": len(dev),
        "test_cases": len(test),
        "dev_first_date": min(dev_dates) if dev_dates else "",
        "dev_last_date": dev_last,
        "test_first_date": test_first,
        "test_last_date": max(test_dates) if test_dates else "",
        "dev_accidents": len(dev_accidents),
        "test_accidents": len(test_accidents),
        "split_accident_overlap": sorted(dev_accidents & test_accidents),
        "split_group_overlap": sorted(dev_groups & test_groups),
        "split_date_overlap": sorted(dev_dates & test_dates),
        "strict_chronological_boundary": "PASS" if chronological else "FAIL",
        "dev_action_distribution": distribution(dev, "gold_action"),
        "test_action_distribution": distribution(test, "gold_action"),
        "dev_resource_type_distribution": distribution(dev, "resource_type"),
        "test_resource_type_distribution": distribution(test, "resource_type"),
    }
    stats["temporal_split_status"] = split_status(stats)

    if cutoff is not None:
        if stats["split_date_overlap"] or not chronological:
            raise ValueError("Temporal split chronology invariant violated.")
        if stats["split_accident_overlap"] or stats["split_group_overlap"]:
            raise ValueError("Temporal split grouping invariant violated.")
    return dev, test, stats


def main():
    parser = argparse.ArgumentParser(description="Build static and chronological benchmark splits from final annotations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be between 0 and 1")
    dev, test, _ = build_static_split(seed=args.seed, dev_fraction=args.dev_fraction)
    csv_write(STATIC_DIR / "dev_pairs.csv", dev, PAIR_FIELDS)
    csv_write(STATIC_DIR / "test_pairs.csv", test, PAIR_FIELDS)
    _, cases, _ = build_dynamic_cases()
    dynamic_dev, dynamic_test, stats = build_temporal_split(cases)
    if stats["temporal_split_status"] == "FAIL":
        raise ValueError("Cannot form a valid chronological split")
    csv_write(DYNAMIC_DIR / "dynamic_dev.csv", dynamic_dev, SPLIT_FIELDS)
    csv_write(DYNAMIC_DIR / "dynamic_test.csv", dynamic_test, SPLIT_FIELDS)
    print(f"Static dev/test: {len(dev)}/{len(test)}; dynamic dev/test: {len(dynamic_dev)}/{len(dynamic_test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
