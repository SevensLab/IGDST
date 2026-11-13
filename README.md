# IGDST: Resource-specific safety knowledge extraction

IGDST extracts resource-specific safety knowledge from industrial accident narratives. For each resource mentioned in an accident, it assigns controlled labels in three dimensions:

- hazards;
- operation stages;
- accident types.

The framework supports both static tagging and chronological knowledge-profile updates. In the dynamic setting, IGDST compares labels supported by a new accident with the resource's earlier profile and returns either `ADD` with genuinely new labels or `NO_CHANGE`.

## Repository contents

This public package is intentionally limited to the materials needed to inspect and reuse the method.

| Path | Contents |
| --- | --- |
| `scripts/` | Preprocessing, split construction, static and dynamic inference, and evaluation code |
| `prompts/` | Resource extraction, tagging, generation, and evidence-verification prompts |
| `config/tag_definitions_en_v1.json` | Controlled vocabulary and label definitions |
| `config/keyword_rules_v1.json` | Rules for the offline keyword baseline |
| `config/resource_rules_v1.json` | Resource extraction and normalization rules |
| `config/llm_only_mapping_v1.json` | Mapping from unconstrained outputs to controlled labels |
| `config/datasets.json` | Dataset identifiers and expected source filenames |
| `examples/accident_example.csv` | One synthetic accident used for an offline demonstration |

The repository contains no experimental predictions, result tables, gold-standard annotations, source datasets, model responses, API credentials, service endpoints, or manuscript files.

## Method outline

1. `preprocess.py` cleans accident narratives, extracts candidate resources, and normalizes resource names.
2. `prepare.py` constructs grouped static splits and chronological dynamic cases from author-supplied annotations.
3. `static.py` implements keyword, embedding, unconstrained LLM, vocabulary-constrained LLM, and IGDST tagging methods.
4. `dynamic.py` implements rule-based and profile-aware update methods.
5. `evaluate.py` computes static and dynamic metrics and accident-cluster bootstrap intervals.
6. `common.py` contains shared readers, validation, inference-adapter, and metric utilities.

IGDST separates label proposal from evidence verification. Candidate labels must belong to the controlled vocabulary, be supported by an accident evidence span, and satisfy resource-specific checks. Dynamic updates then remove labels already present in the earlier resource profile.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

`sentence-transformers` is needed only for the embedding baseline. The offline rule example does not require an inference service.

## Offline example

Run the included synthetic example without making external model calls:

```bash
IGDST_RESULTS_DIR=/tmp/igdst_example \
python scripts/preprocess.py \
  --source examples/accident_example.csv \
  --mode rules
```

The example contains one invented accident narrative. It contains no reference labels, model predictions, or paper results.

## Expected private data interface

Full benchmark preparation requires author-supplied workbooks outside this repository:

```text
data/gold/ihms.xlsx
data/gold/osha.xlsx
```

Each workbook uses three sheets:

- `Annotation`: resource–accident pairs and the four gold fields;
- `Accident_Context`: complete accident narratives;
- `Tag_Vocabulary`: allowed tags and definitions.

The four annotation fields are `gold_include_resource`, `gold_hazard_tags`, `gold_operation_stage_tags`, and `gold_accident_type_tags`. Multilabel values use semicolon-separated controlled tags.

The data and output roots can be supplied outside the repository with `IGDST_DATA_DIR` and `IGDST_RESULTS_DIR`. Generated predictions and metrics should remain in that external output location.

## Inference configuration

The repository includes the prompts and a provider-neutral inference adapter, but it deliberately includes no endpoint, model selection, key, or provider-specific configuration. Anyone running model-based methods must provide their own private configuration outside the repository. The offline preprocessing and rule baselines do not require it.

## Reproducibility boundary

The public materials expose the algorithms, prompts, controlled vocabulary, rules, data schema, and evaluation implementation. Reproducing the manuscript's numerical results additionally requires the original datasets, completed expert annotations, and inference outputs, which are not distributed in this repository.
