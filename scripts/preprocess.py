"""Raw CSV preparation, resource extraction and conservative normalization."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from common import (ROOT, DATASET, DATA_DIR, RESULTS_DIR, csv_write, json_read, jsonl_read,
                    jsonl_write, call_llm_json, llm_configured)
RESOURCE_TYPES = {"chemical", "equipment", "consumable", "other"}
read_jsonl = jsonl_read

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_date(value: str) -> tuple[str, datetime | None]:
    value = normalize_whitespace(value)
    if not value:
        return "", None
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.date().isoformat(), dt
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(value)
        return dt.date().isoformat(), dt
    except ValueError:
        return value, None


def lexical_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}
BASE_COLUMNS = [
    "accident_id",
    "date",
    "country",
    "local",
    "industry_sector",
    "accident_level",
    "potential_accident_level",
    "genre",
    "employee_type",
    "critical_risk",
    "description_raw",
    "description_clean",
    "description_hash",
    "duplicate_group_id",
    "is_duplicate_description",
    "usable_for_llm",
]


EXTERNAL_COLUMNS = [
    "source_dataset",
    "source_record_id",
    "source_title",
    "source_location",
    "source_metadata_json",
]


OUTPUT_COLUMNS = BASE_COLUMNS + EXTERNAL_COLUMNS


DROPPED_COLUMNS = OUTPUT_COLUMNS + ["filter_profile", "filter_reason"]




OSHA_CHEMICAL_RE = re.compile(
    r"\b(chemical|acid|alkali|caustic|corrosive|toxic|noxious|allergen|vapor|vapour|gas|fume|"
    r"solvent|chlorine|ammonia|sulfide|sulfur|oil|fuel|gasoline|diesel|paint|pesticide|cement)\b"
)


OSHA_FIRE_THERMAL_PRESSURE_RE = re.compile(
    r"\b(fire|flame|explosion|explosive|ignition|burn|hot|thermal|steam|weld|welding|torch|"
    r"pressuri[sz]ed|pressure|compressed air|rupture|burst|discharged object|discharged substance)\b"
)


OSHA_ELECTRICAL_RE = re.compile(
    r"\b(electric|electrical|electricity|electrocution|shock|power line|transformer|switchboard|"
    r"switches|fuses|power cord|extension cord)\b"
)


OSHA_STRONG_EVENT_RE = re.compile(
    r"\b(caught in running equipment|caught, entangled in running powered equipment|"
    r"struck against moving part of machinery|struck by running powered equipment|"
    r"ignition of vapors|direct exposure to electricity|contact with hot objects or substances|"
    r"inhalation of harmful substance|exposure through intact skin|struck by discharged object or substance)\b"
)


OSHA_CONCRETE_SOURCE_RE = re.compile(
    r"\b(forklift|pallet jack|truck|vehicle|tractor|loader|crane|hoist|conveyor|saw|press|"
    r"grinder|drill|lathe|machinery|rig|power line|transformer|switchboard|switches|fuses|"
    r"power cord|extension cord|pipe|duct|tubing|hose|valve|tank|vat|vessel|pump|torch|"
    r"welder|welding|nail gun|jack|blade|belt|chain|motor|machine guard|aerial lift|"
    r"scissor lift|ladder|scaffold|trailer|cart|dolly|hand truck|door|gate|rack|pallet|"
    r"beam|metal|glass)\b"
)


OSHA_RESOURCE_FALL_RE = re.compile(
    r"\b(ladder|scaffold|aerial lift|scissor lift|boom lift|platform|roof|floor opening|"
    r"existing opening|loading dock|dock plate)\b"
)


OSHA_FALL_RE = re.compile(r"\b(fall|fell|slip|slipping|trip|tripping)\b")


OSHA_PERSON_CONTEXT_RE = re.compile(
    r"\b(hitting|kicking|beating|shoving|shooting|gunshot|physical contact with person|patient)\b"
)


OSHA_BODY_ONLY_RE = re.compile(
    r"\b(overexertion|bodily motion|bodily reaction|heart attack|myocardial infarction|"
    r"environmental heat|heat exhaustion|heat prostration|heat fatigue)\b"
)


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def infer_dataset(path: Path, columns: list[str]) -> str:
    column_set = set(columns)
    if {"Data", "Countries", "Industry Sector", "Critical Risk", "Description"}.issubset(column_set):
        return "ihms"
    if {"ID", "EventDate", "Final Narrative", "NatureTitle", "EventTitle", "SourceTitle"}.issubset(column_set):
        return "osha_severe_injuries_2015_2025"
    raise ValueError(f"Unsupported external CSV schema: {path}")


def first_nonempty(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = normalize_whitespace(row.get(key, ""))
        if value:
            return value
    return ""


def location_join(*parts: str) -> str:
    return ", ".join(part for part in (normalize_whitespace(p) for p in parts) if part)


def compact_filter_text(row: dict[str, str]) -> str:
    fields = [
        "Critical Risk",
        "Description",
        "EventTitle",
        "NatureTitle",
        "SourceTitle",
        "Secondary Source Title",
        "Final Narrative",
    ]
    return " ".join(normalize_whitespace(row.get(field, "")).lower() for field in fields)


def ihms_filter_decision(row: dict[str, str]) -> tuple[bool, str]:
    """Keep IHMS incidents with a concrete IGDST resource or energy signal."""
    text = compact_filter_text(row)
    risk = normalize_whitespace(row.get("Critical Risk", "")).lower()

    if re.search(r"\b(chemical|acid|alkali|caustic|sulfide|sulphide|solution|gas|fume|oil|fuel|solvent)\b", text):
        return True, "keep_chemical_or_material_exposure"
    if re.search(r"\b(fire|flame|explosion|explosive|ignition|burn|hot|thermal|steam|pressure|pressurized|pressurised|compressed)\b", text):
        return True, "keep_fire_thermal_pressure"
    if re.search(r"\b(electric|electrical|electricity|electrocution|shock|energized|voltage|substation|transformer)\b", text):
        return True, "keep_electrical"
    if re.search(r"\b(drill|jumbo|pump|pipe|piping|hose|valve|tank|conveyor|crusher|grinder|crane|hoist|forklift|truck|vehicle|machine|equipment|beam|bar|ladder|scaffold|rock|tool)\b", text):
        return True, "keep_equipment_or_material_event"
    if re.search(r"\b(fall|fell|slip|trip|stumble)\b", text) and re.search(r"\b(ladder|scaffold|platform|height|roof|opening|jumbo|truck|vehicle|equipment)\b", text):
        return True, "keep_resource_linked_fall"
    if re.search(r"\b(fall|fell|slip|trip|stumble|overexertion|bodily motion|heat exhaustion|heart attack)\b", risk + " " + text):
        return False, "exclude_fall_or_body_only_without_resource_signal"
    return False, "exclude_no_core_resource_signal"


def core_osha_filter_decision(row: dict[str, str]) -> tuple[bool, str]:
    text = compact_filter_text(row)
    event = normalize_whitespace(row.get("EventTitle", "")).lower()
    source_text = " ".join(
        [
            normalize_whitespace(row.get("SourceTitle", "")).lower(),
            normalize_whitespace(row.get("Secondary Source Title", "")).lower(),
        ]
    )

    if OSHA_PERSON_CONTEXT_RE.search(text) and not OSHA_CONCRETE_SOURCE_RE.search(source_text):
        return False, "exclude_person_or_patient_context"

    reasons = []
    if OSHA_CHEMICAL_RE.search(text):
        reasons.append("keep_chemical_or_material_exposure")
    if OSHA_FIRE_THERMAL_PRESSURE_RE.search(text) and "environmental heat" not in text:
        reasons.append("keep_fire_thermal_pressure")
    if OSHA_ELECTRICAL_RE.search(text):
        reasons.append("keep_electrical")
    if OSHA_STRONG_EVENT_RE.search(event):
        reasons.append("keep_strong_equipment_or_exposure_event")
    if OSHA_CONCRETE_SOURCE_RE.search(source_text) and not (OSHA_FALL_RE.search(event) and not OSHA_RESOURCE_FALL_RE.search(source_text)):
        reasons.append("keep_concrete_source_equipment_or_material")
    if OSHA_FALL_RE.search(event) and OSHA_RESOURCE_FALL_RE.search(source_text):
        reasons.append("keep_resource_linked_fall")

    if reasons:
        return True, ";".join(reasons)
    if OSHA_FALL_RE.search(event):
        return False, "exclude_fall_slip_trip_without_resource_signal"
    if OSHA_BODY_ONLY_RE.search(text):
        return False, "exclude_body_motion_or_environmental_health"
    return False, "exclude_no_core_resource_signal"


def filter_decision(dataset: str, row: dict[str, str], profile: str) -> tuple[bool, str]:
    if profile in {"", "none"}:
        return True, "no_filter"
    if profile not in {"ihms_igdst_core_v1", "osha_igdst_core_v1"}:
        raise ValueError(f"Unsupported filter profile: {profile}")
    if dataset == "ihms":
        return ihms_filter_decision(row) if profile == "ihms_igdst_core_v1" else (True, "profile_not_applicable")
    if dataset != "osha_severe_injuries_2015_2025":
        return True, "profile_not_applicable"
    return core_osha_filter_decision(row)


def adapt_row(dataset: str, row: dict[str, str], source_index: int) -> dict[str, Any]:
    if dataset == "ihms":
        date_text, date_dt = parse_date(row.get("Data", ""))
        metadata = {
            "critical_risk": first_nonempty(row, "Critical Risk"),
            "source_index": str(source_index),
        }
        return {
            "_source_index": source_index,
            "_date_sort": date_dt,
            "date": date_text,
            "country": first_nonempty(row, "Countries"),
            "local": first_nonempty(row, "Local"),
            "industry_sector": first_nonempty(row, "Industry Sector"),
            "accident_level": first_nonempty(row, "Accident Level"),
            "potential_accident_level": first_nonempty(row, "Potential Accident Level"),
            "genre": first_nonempty(row, "Genre"),
            "employee_type": first_nonempty(row, "Employee or Third Party"),
            "critical_risk": first_nonempty(row, "Critical Risk"),
            "description_raw": row.get("Description", ""),
            "source_dataset": dataset,
            "source_record_id": first_nonempty(row, "", "Unnamed: 0"),
            "source_title": "IHMStefanini",
            "source_location": location_join(row.get("Countries", ""), row.get("Local", "")),
            "source_metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        }
    if dataset == "osha_severe_injuries_2015_2025":
        date_text, date_dt = parse_date(row.get("EventDate", ""))
        description_raw = row.get("Final Narrative", "")
        local = location_join(row.get("City", ""), row.get("State", ""))
        metadata = {
            "employer": first_nonempty(row, "Employer"),
            "nature": first_nonempty(row, "NatureTitle"),
            "event": first_nonempty(row, "EventTitle"),
            "source": first_nonempty(row, "SourceTitle"),
            "secondary_source": first_nonempty(row, "Secondary Source Title"),
            "hospitalized": first_nonempty(row, "Hospitalized"),
            "amputation": first_nonempty(row, "Amputation"),
            "loss_of_eye": first_nonempty(row, "Loss of Eye"),
            "latitude": first_nonempty(row, "Latitude"),
            "longitude": first_nonempty(row, "Longitude"),
            "federal_state": first_nonempty(row, "FederalState"),
        }
        return {
            "_source_index": source_index,
            "_date_sort": date_dt,
            "date": date_text,
            "country": "United States",
            "local": local,
            "industry_sector": first_nonempty(row, "Primary NAICS"),
            "accident_level": first_nonempty(row, "NatureTitle"),
            "potential_accident_level": "",
            "genre": "",
            "employee_type": "",
            "critical_risk": first_nonempty(row, "EventTitle"),
            "description_raw": description_raw,
            "source_dataset": dataset,
            "source_record_id": first_nonempty(row, "ID"),
            "source_title": first_nonempty(row, "Employer"),
            "source_location": location_join(row.get("Address1", ""), row.get("Address2", ""), local, row.get("Zip", "")),
            "source_metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def clean_rows(
    dataset: str,
    raw_rows: list[dict[str, str]],
    filter_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    staged = []
    dropped = []
    filter_counts: Counter[str] = Counter()
    for idx, row in enumerate(raw_rows):
        item = adapt_row(dataset, row, idx)
        description_clean = normalize_whitespace(item["description_raw"])
        item["description_clean"] = description_clean
        item["description_hash"] = hashlib.sha256(description_clean.encode("utf-8")).hexdigest() if description_clean else ""
        item["usable_for_llm"] = str(bool(description_clean)).lower()
        keep, filter_reason = filter_decision(dataset, row, filter_profile)
        filter_counts.update([filter_reason])
        item["filter_profile"] = filter_profile
        item["filter_reason"] = filter_reason
        if not keep:
            dropped.append(item)
            continue
        staged.append(item)

    staged.sort(key=lambda item: (item["_date_sort"] is None, item["_date_sort"] or "", item["_source_index"]))
    hash_counts = Counter(row["description_hash"] for row in staged if row["description_hash"])
    group_ids: dict[str, str] = {}
    group_counter = 0
    for row in staged:
        desc_hash = row["description_hash"]
        if desc_hash and hash_counts[desc_hash] > 1 and desc_hash not in group_ids:
            group_counter += 1
            group_ids[desc_hash] = f"DUP_{group_counter:04d}"

    cleaned = []
    prefix = {"ihms": "IHMS", "osha_severe_injuries_2015_2025": "OSHA"}[dataset]
    for idx, row in enumerate(staged, start=1):
        desc_hash = row.get("description_hash", "")
        row["duplicate_group_id"] = group_ids.get(desc_hash, "")
        row["is_duplicate_description"] = str(bool(desc_hash and hash_counts[desc_hash] > 1)).lower()
        out = {key: row.get(key, "") for key in OUTPUT_COLUMNS if key != "accident_id"}
        out = {"accident_id": f"{prefix}_{idx:06d}", **out}
        cleaned.append(out)
    dropped.sort(key=lambda item: (item["_date_sort"] is None, item["_date_sort"] or "", item["_source_index"]))
    dropped_hash_counts = Counter(row["description_hash"] for row in dropped if row["description_hash"])
    dropped_out = []
    for idx, row in enumerate(dropped, start=1):
        desc_hash = row.get("description_hash", "")
        row["duplicate_group_id"] = ""
        row["is_duplicate_description"] = str(bool(desc_hash and dropped_hash_counts[desc_hash] > 1)).lower()
        out = {key: row.get(key, "") for key in DROPPED_COLUMNS if key != "accident_id"}
        out = {"accident_id": f"{prefix}_DROPPED_{idx:06d}", **out}
        dropped_out.append(out)
    return cleaned, dropped_out, filter_counts


def sample_clean_rows(
    cleaned: list[dict[str, Any]],
    sample_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sample_size <= 0 or sample_size >= len(cleaned):
        return cleaned, []

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in cleaned:
        year = str(row.get("date", ""))[:4] or "unknown"
        groups.setdefault(year, []).append(row)
    years = sorted(groups.keys())
    total = sum(len(rows) for rows in groups.values())
    quotas = {}
    assigned = 0
    remainders = []
    for year in years:
        exact = sample_size * len(groups[year]) / total
        quota = min(len(groups[year]), int(math.floor(exact)))
        quotas[year] = quota
        assigned += quota
        remainders.append((exact - quota, year))
    for _, year in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        if quotas[year] < len(groups[year]):
            quotas[year] += 1
            assigned += 1

    selected = []
    sampled_out = []
    for year in years:
        rows = groups[year]
        quota = quotas[year]
        if quota >= len(rows):
            selected.extend(rows)
            continue
        if quota <= 0:
            sampled_out.extend([{**row, "sample_strategy": "year_stratified"} for row in rows])
            continue
        if quota == 1:
            keep_indexes = {len(rows) // 2}
        else:
            keep_indexes = {
                round(i * (len(rows) - 1) / (quota - 1))
                for i in range(quota)
            }
        for idx, row in enumerate(rows):
            if idx in keep_indexes:
                selected.append(row)
            else:
                sampled_out.append({**row, "sample_strategy": "year_stratified"})

    selected.sort(key=lambda item: (item.get("_date_sort") is None, item.get("_date_sort") or "", item.get("_source_index", 0)))
    sampled_out.sort(key=lambda item: (item.get("_date_sort") is None, item.get("_date_sort") or "", item.get("_source_index", 0)))
    return selected, sampled_out
@dataclass(frozen=True)
class RuleExtraction:
    resources: list[dict[str, Any]]
    needs_llm: bool
    llm_reason: str
    rule_version: str


def load_rule_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("version") or not isinstance(config.get("rules"), list):
        raise ValueError(f"Invalid resource rule configuration: {path}")
    return config


@lru_cache(maxsize=None)
def term_pattern(term: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in term.strip().split()]
    body = r"[\s-]+".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", flags=re.IGNORECASE)


def overlaps_excluded(start: int, end: int, excluded_spans: list[tuple[int, int]]) -> bool:
    return any(start < excluded_end and end > excluded_start for excluded_start, excluded_end in excluded_spans)


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" \t\r\n.,;:()[]{}")


def extract_rule_resources(description: str, config: dict[str, Any]) -> RuleExtraction:
    excluded_spans = []
    for phrase in config.get("excluded_phrases", []):
        excluded_spans.extend((match.start(), match.end()) for match in term_pattern(phrase).finditer(description))

    matches = []
    for rule in config["rules"]:
        for term in rule.get("terms", []):
            for match in term_pattern(term).finditer(description):
                if overlaps_excluded(match.start(), match.end(), excluded_spans):
                    continue
                evidence = match.group(0).strip()
                matches.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "raw_name": evidence,
                        "resource_type": rule["resource_type"],
                        "evidence_span": evidence,
                        "confidence": float(rule["confidence"]),
                        "extraction_source": "rule",
                        "rule_id": rule["id"],
                        "uncertain": bool(rule.get("uncertain", False)),
                        "generic": bool(rule.get("generic", False)),
                    }
                )

    specific_types = {item["resource_type"] for item in matches if not item["generic"]}
    matches = [item for item in matches if not item["generic"] or item["resource_type"] not in specific_types]
    matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), -item["confidence"]))

    kept = []
    seen = set()
    for item in matches:
        key = (item["resource_type"], normalize_key(item["raw_name"]))
        if key in seen:
            continue
        contained = any(
            item["resource_type"] == existing["resource_type"]
            and item["start"] >= existing["start"]
            and item["end"] <= existing["end"]
            for existing in kept
        )
        if contained:
            continue
        kept.append(item)
        seen.add(key)

    resources = []
    uncertain = False
    for item in kept:
        uncertain = uncertain or item["uncertain"]
        resources.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"start", "end", "generic"}
            }
        )
        resources[-1]["rule_uncertain"] = resources[-1].pop("uncertain")

    if not resources and config.get("fallback_on_no_resource", True):
        needs_llm = True
        reason = "no_rule_match"
    elif uncertain and config.get("fallback_on_uncertain", True):
        needs_llm = True
        reason = "uncertain_rule_match"
    else:
        needs_llm = False
        reason = ""
    return RuleExtraction(resources, needs_llm, reason, str(config["version"]))


def annotate_llm_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **copy.deepcopy(resource),
            "extraction_source": "llm",
            "rule_id": "",
        }
        for resource in resources
    ]


def merge_resources(rule_resources: list[dict[str, Any]], llm_resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    confirmed_llm = annotate_llm_resources(llm_resources)
    llm_keys = {
        (str(resource.get("resource_type", "")), normalize_key(str(resource.get("raw_name", ""))))
        for resource in confirmed_llm
    }
    confirmed_rules = [
        resource
        for resource in rule_resources
        if not resource.get("rule_uncertain")
        or (str(resource.get("resource_type", "")), normalize_key(str(resource.get("raw_name", "")))) in llm_keys
    ]
    for resource in [*confirmed_rules, *confirmed_llm]:
        key = (str(resource.get("resource_type", "")), normalize_key(str(resource.get("raw_name", ""))))
        if not key[1]:
            continue
        existing = by_key.get(key)
        if existing is None:
            item = copy.deepcopy(resource)
            by_key[key] = item
            merged.append(item)
            continue
        existing["confidence"] = max(float(existing.get("confidence", 0)), float(resource.get("confidence", 0)))
        sources = {str(existing.get("extraction_source", "")), str(resource.get("extraction_source", ""))}
        existing["extraction_source"] = "+".join(sorted(source for source in sources if source))
        if not existing.get("evidence_span") and resource.get("evidence_span"):
            existing["evidence_span"] = resource["evidence_span"]
    return merged
PAIR_COLUMNS = [
    "accident_id",
    "date",
    "resource_id",
    "resource_raw",
    "normalized_surface_name",
    "resource_specific_name",
    "resource_canonical",
    "resource_family",
    "resource_type",
    "resource_role",
    "role_reason",
    "eligible_for_tagging",
    "description",
    "evidence_span",
    "extraction_confidence",
    "qa_flag",
]


DICT_COLUMNS = [
    "resource_id",
    "canonical_name",
    "resource_family",
    "resource_type",
    "aliases",
    "mention_count",
    "accident_count",
    "first_seen_date",
    "last_seen_date",
]




PREFIX = {"chemical": "CHEM", "equipment": "EQUIP", "consumable": "CONS", "other": "OTHER"}


SPELLING = {
    "sulphur": "sulfur",
    "sulphide": "sulfide",
    "sulphuric": "sulfuric",
    "aluminium": "aluminum",
}


EQUIPMENT_CANONICAL = [
    (re.compile(r"\b(water |process |sodium sulfide |sodium sulphide |hydraulic |electric )?pumps?\b"), "pump"),
    (re.compile(r"\b(pipes?|piping|pipelines?)\b"), "pipe/piping"),
    (re.compile(r"\bhoses?\b"), "hose"),
    (re.compile(r"\bvalves?\b"), "valve"),
    (re.compile(r"\b(drilling machines?|drills?)\b"), "drill"),
    (re.compile(r"\b(trucks?|vehicles?|loaders?|forklifts?|tractors?)\b"), "vehicle/mobile equipment"),
    (re.compile(r"\b(tanks?|vessels?)\b"), "tank/vessel"),
    (re.compile(r"\bconveyors?\b"), "conveyor"),
    (re.compile(r"\bgrinders?\b"), "grinder"),
    (re.compile(r"\bcranes?\b"), "crane"),
]


CHEMICAL_CANONICAL = [
    (re.compile(r"\bso2\b|\bsulfur dioxide\b"), "sulfur dioxide"),
    (re.compile(r"\bh2so4\b|\bsulfuric acid\b"), "sulfuric acid"),
    (re.compile(r"\bsodium sulfide\b|\bsodium sulphide\b"), "sodium sulfide"),
]


MITIGATION_TERMS = {
    "emergency shower",
    "shower",
    "eyewash",
    "eye wash",
    "first aid",
    "hospital",
    "medical center",
    "medical centre",
    "clinic",
    "ambulatory",
    "ambulance",
    "doctor",
    "infirmary",
}


PPE_TERMS = {
    "helmet",
    "glove",
    "gloves",
    "safety shoe",
    "safety shoes",
    "goggle",
    "goggles",
    "mask",
    "respirator",
    "earplug",
    "earplugs",
    "ppe",
}


PERSON_OR_BODY_TERMS = {
    "employee",
    "worker",
    "operator",
    "mechanic",
    "supervisor",
    "collaborator",
    "assistant",
    "hand",
    "left hand",
    "right hand",
    "finger",
    "fingers",
    "face",
    "foot",
    "eye",
    "arm",
    "head",
    "leg",
}


def load_accidents(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["accident_id"]: row for row in csv.DictReader(f)}


def singularize_light(text: str) -> str:
    words = []
    for word in text.split():
        irregular = {
            "gloves": "glove",
            "pipes": "pipe",
            "hoses": "hose",
            "valves": "valve",
            "boxes": "box",
            "buses": "bus",
            "gases": "gas",
        }
        if word in irregular:
            word = irregular[word]
        elif len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("ches"):
            word = word[:-2]
        elif len(word) > 5 and word.endswith("shes"):
            word = word[:-2]
        elif len(word) > 5 and word.endswith("xes"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def level1_normalize(raw_name: str) -> str:
    text = raw_name.lower().strip()
    for src, dst in SPELLING.items():
        text = text.replace(src, dst)
    text = re.sub(r"[/_,;:()]+", " ", text)
    text = re.sub(r"[^a-z0-9+\-. ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return singularize_light(text)


def canonicalize(raw_name: str, resource_type: str) -> tuple[str, str, str]:
    base = level1_normalize(raw_name)
    patterns = EQUIPMENT_CANONICAL if resource_type == "equipment" else CHEMICAL_CANONICAL if resource_type == "chemical" else []
    for pattern, canonical in patterns:
        if pattern.search(base):
            return base, base, canonical
    return base, base, base


def infer_resource_role(resource: dict[str, Any], resource_type: str, canonical: str, description: str) -> tuple[str, str, bool]:
    raw = level1_normalize(resource.get("raw_name", ""))
    evidence = level1_normalize(resource.get("evidence_span", ""))
    text = level1_normalize(description)
    haystack = " ".join([raw, canonical, evidence])
    raw_tokens = set(raw.split())
    canonical_tokens = set(canonical.split())

    if any(term in haystack for term in MITIGATION_TERMS):
        return "mitigation", "post-incident response or medical/cleanup facility", False
    if any(term in haystack for term in PPE_TERMS):
        return "mitigation", "PPE or protective equipment excluded from dynamic hazard tagging", False
    if any(term == raw or term == canonical for term in PERSON_OR_BODY_TERMS):
        return "context", "person/body/location-like entity excluded from tagging", False
    if (raw_tokens | canonical_tokens) & {term for term in PERSON_OR_BODY_TERMS if " " not in term}:
        return "context", "person/body/location-like entity excluded from tagging", False
    if resource_type == "other":
        return "context", "resource_type other excluded from tagging", False

    risk_text = " ".join([evidence, text])
    failed_patterns = [
        "failed",
        "failure",
        "broke",
        "broken",
        "rupture",
        "ruptured",
        "uncoupled",
        "disconnect",
        "detached",
        "loosen",
        "leak",
        "leaked",
        "burst",
        "slid",
        "projected",
    ]
    handled_patterns = [
        "removing",
        "handling",
        "carrying",
        "lifting",
        "pull",
        "pushing",
        "loading",
        "unloading",
        "cleaning",
        "maintenance",
        "repair",
        "activation",
        "operating",
    ]
    hazard_patterns = [
        "splash",
        "splashed",
        "projected",
        "sprayed",
        "burn",
        "hot",
        "acid",
        "sulfide",
        "sulfur",
        "gas",
        "chemical",
        "solution",
        "oil",
        "water",
        "pressure",
        "pressurized",
    ]

    if resource_type == "chemical":
        if any(pattern in risk_text for pattern in hazard_patterns):
            return "hazard_source", "chemical/material is tied to exposure, splash, burn, release, or pressure evidence", True
        return "involved_material", "chemical/material appears in the accident operation", True
    if resource_type == "equipment":
        if any(pattern in risk_text for pattern in failed_patterns):
            return "failed_component", "equipment/component is tied to failure, release, abnormal movement, or leakage evidence", True
        if any(pattern in risk_text for pattern in handled_patterns):
            return "handled_object", "equipment/object is being handled, operated, cleaned, maintained, loaded, or repaired", True
        return "involved_material", "equipment appears in accident context and is eligible for conservative candidate tagging", True
    if resource_type == "consumable":
        return "handled_object", "consumable appears in the accident operation", True
    return "context", "not eligible for dynamic safety tagging", False


def build_outputs(
    accidents_path: Path,
    mentions_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accidents = load_accidents(accidents_path)
    mentions = read_jsonl(mentions_path)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    pair_candidates = []

    for record in mentions:
        accident_id = record.get("accident_id", "")
        accident = accidents.get(accident_id, {})
        for resource in record.get("resources", []):
            resource_type = resource.get("resource_type", "other")
            normalized_surface, specific_name, resource_family = canonicalize(resource.get("raw_name", ""), resource_type)
            canonical = resource_family
            if not canonical:
                continue
            key = (resource_type, canonical)
            grouped.setdefault(
                key,
                {
                    "canonical_name": canonical,
                    "resource_type": resource_type,
                    "aliases": set(),
                    "mention_count": 0,
                    "accidents": set(),
                    "dates": [],
                },
            )
            grouped[key]["aliases"].add(level1_normalize(resource.get("raw_name", "")))
            grouped[key]["mention_count"] += 1
            grouped[key]["accidents"].add(accident_id)
            if accident.get("date"):
                grouped[key]["dates"].append(accident["date"])
            role, role_reason, eligible_for_tagging = infer_resource_role(
                resource,
                resource_type,
                canonical,
                accident.get("description_clean", ""),
            )
            pair_candidates.append(
                {
                    "accident_id": accident_id,
                    "date": accident.get("date", ""),
                    "resource_type": resource_type,
                    "resource_raw": resource.get("raw_name", ""),
                    "normalized_surface_name": normalized_surface,
                    "resource_specific_name": specific_name,
                    "resource_canonical": canonical,
                    "resource_family": resource_family,
                    "resource_role": role,
                    "role_reason": role_reason,
                    "eligible_for_tagging": str(eligible_for_tagging).lower(),
                    "description": accident.get("description_clean", ""),
                    "evidence_span": resource.get("evidence_span", ""),
                    "extraction_confidence": resource.get("confidence", ""),
                    "qa_flag": resource.get("qa_flag", ""),
                }
            )

    dictionary = []
    id_by_key = {}
    counters = defaultdict(int)
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        resource_type, canonical = key
        counters[resource_type] += 1
        resource_id = f"{PREFIX.get(resource_type, 'OTHER')}_{counters[resource_type]:04d}"
        id_by_key[key] = resource_id
        item = grouped[key]
        dates = sorted(item["dates"])
        dictionary.append(
            {
                "resource_id": resource_id,
                "canonical_name": canonical,
                "resource_family": canonical,
                "resource_type": resource_type,
                "aliases": "; ".join(sorted(a for a in item["aliases"] if a)),
                "mention_count": item["mention_count"],
                "accident_count": len(item["accidents"]),
                "first_seen_date": dates[0] if dates else "",
                "last_seen_date": dates[-1] if dates else "",
            }
        )

    seen_pairs = set()
    pairs = []
    for pair in pair_candidates:
        key = (pair["resource_type"], pair["resource_canonical"])
        resource_id = id_by_key.get(key, "")
        dedupe_key = (pair["accident_id"], resource_id)
        if dedupe_key in seen_pairs:
            continue
        seen_pairs.add(dedupe_key)
        pairs.append({"resource_id": resource_id, **pair})

    frequency = []
    for item in dictionary:
        frequency.append(
            {
                "resource_type": item["resource_type"],
                "canonical_name": item["canonical_name"],
                "resource_id": item["resource_id"],
                "accident_count": item["accident_count"],
                "mention_count": item["mention_count"],
                "repeat_ge_2": str(int(item["accident_count"]) >= 2).lower(),
                "repeat_ge_3": str(int(item["accident_count"]) >= 3).lower(),
                "repeat_ge_5": str(int(item["accident_count"]) >= 5).lower(),
                "repeat_ge_10": str(int(item["accident_count"]) >= 10).lower(),
            }
        )
    frequency.sort(key=lambda row: (row["resource_type"], -int(row["accident_count"]), row["canonical_name"]))
    return dictionary, pairs, frequency
def normalize_resource_payload(accident_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    resources = parsed.get("resources", [])
    if not isinstance(resources, list):
        resources = []
    clean_resources = []
    for item in resources:
        if not isinstance(item, dict):
            continue
        resource_type = str(item.get("resource_type", "")).strip().lower()
        if resource_type not in RESOURCE_TYPES:
            resource_type = "other"
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        clean_resources.append(
            {
                "raw_name": str(item.get("raw_name", "")).strip(),
                "resource_type": resource_type,
                "evidence_span": str(item.get("evidence_span", "")).strip(),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return {"accident_id": accident_id, "resources": clean_resources}


def build_messages(prompt_text: str, accident_id: str, description: str) -> list[dict[str, str]]:
    user_payload = {"accident_id": accident_id, "description": description}
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
BODY_PARTS = {
    "finger",
    "fingers",
    "hand",
    "hands",
    "foot",
    "feet",
    "eye",
    "eyes",
    "arm",
    "arms",
    "head",
    "leg",
    "legs",
    "face",
    "neck",
    "back",
    "shoulder",
    "knee",
}


PERSON_TERMS = {"worker", "employee", "operator", "mechanic", "supervisor", "person", "maid", "technician"}


def flags_for_resource(resource: dict[str, Any], description: str) -> list[str]:
    flags = []
    raw_name = str(resource.get("raw_name", ""))
    evidence = str(resource.get("evidence_span", ""))
    resource_type = str(resource.get("resource_type", ""))
    try:
        confidence = float(resource.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1

    if not raw_name or not evidence:
        flags.append("missing_required_field")
    if resource_type not in RESOURCE_TYPES:
        flags.append("invalid_resource_type")
    if confidence < 0 or confidence > 1:
        flags.append("invalid_confidence")
    if evidence and evidence not in description:
        flags.append("unsupported_evidence")
    if raw_name and evidence and not (lexical_tokens(raw_name) & lexical_tokens(evidence)):
        flags.append("weak_lexical_overlap")
    raw_tokens = lexical_tokens(raw_name)
    body_part_exceptions = {"hand tool", "hand tools", "hand truck"}
    if raw_name.lower().strip() not in body_part_exceptions and raw_tokens & BODY_PARTS and resource_type in {"chemical", "equipment", "consumable"}:
        flags.append("suspicious_body_part")
    if raw_tokens & PERSON_TERMS:
        flags.append("suspicious_person")
    return flags


def extract_resources(accidents, mode):
    config = load_rule_config(ROOT / "config/resource_rules_v1.json")
    prompt = (ROOT / "prompts/resource_extraction.txt").read_text(encoding="utf-8")
    if mode != "rules" and not llm_configured():
        raise RuntimeError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL are required for hybrid/llm extraction")
    records = []
    for accident in accidents:
        description = accident["description_clean"]
        rule = extract_rule_resources(description, config)
        resources = rule.resources if mode != "llm" else []
        if mode == "llm" or (mode == "hybrid" and rule.needs_llm):
            parsed, _ = call_llm_json(build_messages(prompt, accident["accident_id"], description))
            candidates = annotate_llm_resources(normalize_resource_payload(accident["accident_id"], parsed)["resources"])
            resources = merge_resources(resources, candidates)
        resources = [{**r, "qa_flag": ";".join(flags_for_resource(r, description))} for r in resources]
        records.append({"accident_id": accident["accident_id"], "resources": resources})
    return records


def main():
    parser = argparse.ArgumentParser(description="Clean raw accident CSVs and extract candidate resources for annotation.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mode", choices=["rules", "hybrid", "llm"], default="rules")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional year-stratified sample; 0 keeps all records")
    parser.add_argument("--filter-profile", default="none", help="none or the explicitly chosen dataset filter")
    args = parser.parse_args()
    config = json_read(ROOT / "config/datasets.json")[DATASET]
    source = args.source or DATA_DIR / "raw" / config["source_file"]
    raw, columns = read_rows(source)
    dataset = infer_dataset(source, columns)
    cleaned, _, _ = clean_rows(dataset, raw, args.filter_profile)
    if args.sample_size < 0:
        parser.error("--sample-size must be nonnegative")
    cleaned, _ = sample_clean_rows(cleaned, args.sample_size)
    output = RESULTS_DIR / "preparation"
    accidents_file = output / "accidents_clean.csv"
    mentions_file = output / "resource_mentions.jsonl"
    csv_write(accidents_file, cleaned, OUTPUT_COLUMNS)
    jsonl_write(mentions_file, extract_resources(cleaned, args.mode))
    dictionary, pairs, _ = build_outputs(accidents_file, mentions_file)
    csv_write(output / "resource_dictionary.csv", dictionary, DICT_COLUMNS)
    csv_write(output / "candidate_pairs.csv", pairs, PAIR_COLUMNS)
    print(f"Prepared {len(cleaned)} accidents and {len(pairs)} candidate pairs in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
