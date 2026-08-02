"""Pure contracts and scoring for controlled counterfactual admissibility pairs."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ADMISSIBILITY_LABELS = ("admissible", "inadmissible", "unknown")
CONDITIONS = ("allow", "block")
CANDIDATE_ROLES = ("focal", "stable_admissible", "stable_inadmissible")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class ProbabilityVector:
    """A validated probability vector with deterministic label ordering."""

    labels: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.labels or len(self.labels) != len(self.values):
            raise ValueError("probability labels and values must be nonempty and aligned")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("probability labels must be unique")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.values):
            raise ValueError("probabilities must be finite and in [0, 1]")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        labels: Sequence[str],
        tolerance: float,
        location: str,
    ) -> ProbabilityVector:
        row = _mapping(value, location)
        expected = tuple(labels)
        if set(row) != set(expected):
            raise ValueError(f"{location} labels differ from {expected!r}")
        vector = cls(
            labels=expected,
            values=tuple(_number(row[label], f"{location}.{label}") for label in expected),
        )
        if abs(sum(vector.values) - 1.0) > tolerance:
            raise ValueError(f"{location} probabilities must sum to 1")
        return vector

    def probability(self, label: str) -> float:
        try:
            return self.values[self.labels.index(label)]
        except ValueError as error:
            raise KeyError(label) from error

    def argmax(self) -> str:
        return self.labels[max(range(len(self.values)), key=self.values.__getitem__)]


@dataclass(frozen=True)
class CounterfactualCandidate:
    """One fixed-pool candidate with scorer-only counterfactual labels."""

    candidate_key: str
    role: str
    text: str
    admissible_allow: bool
    admissible_block: bool

    def __post_init__(self) -> None:
        if not self.candidate_key or not self.text:
            raise ValueError("candidate key and text must be nonempty")
        if self.role not in CANDIDATE_ROLES:
            raise ValueError(f"unexpected candidate role {self.role!r}")
        expected = {
            "focal": (True, False),
            "stable_admissible": (True, True),
            "stable_inadmissible": (False, False),
        }[self.role]
        if (self.admissible_allow, self.admissible_block) != expected:
            raise ValueError(f"candidate role {self.role!r} has inconsistent gold labels")

    def gold(self, condition: str) -> bool:
        if condition == "allow":
            return self.admissible_allow
        if condition == "block":
            return self.admissible_block
        raise ValueError(f"unexpected condition {condition!r}")


@dataclass(frozen=True)
class QueryPair:
    pair_id: str
    allow_query: str
    block_query: str

    def __post_init__(self) -> None:
        if not self.pair_id or not self.allow_query or not self.block_query:
            raise ValueError("query pair fields must be nonempty")
        if self.allow_query == self.block_query:
            raise ValueError("counterfactual queries must differ")


@dataclass(frozen=True)
class CounterfactualScenario:
    scenario_id: str
    axis: str
    source_basis: str
    candidates: tuple[CounterfactualCandidate, ...]
    query_pairs: tuple[QueryPair, ...]

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.axis or not self.source_basis:
            raise ValueError("scenario identity fields must be nonempty")
        if not self.candidates or not self.query_pairs:
            raise ValueError("scenario must contain candidates and query pairs")
        keys = [candidate.candidate_key for candidate in self.candidates]
        roles = [candidate.role for candidate in self.candidates]
        pair_ids = [pair.pair_id for pair in self.query_pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate keys must be unique within a scenario")
        if sorted(roles) != sorted(CANDIDATE_ROLES):
            raise ValueError("scenario must contain exactly one candidate for every role")
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("query pair IDs must be unique within a scenario")


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    scenario_id: str
    axis: str
    source_basis: str
    allow_query: str
    block_query: str
    candidates: tuple[CounterfactualCandidate, ...]

    def case_id(self, condition: str) -> str:
        if condition not in CONDITIONS:
            raise ValueError(f"unexpected condition {condition!r}")
        return f"{self.pair_id}::{condition}"

    def query(self, condition: str) -> str:
        if condition == "allow":
            return self.allow_query
        if condition == "block":
            return self.block_query
        raise ValueError(f"unexpected condition {condition!r}")


def _candidate_from_mapping(value: object, location: str) -> CounterfactualCandidate:
    row = _mapping(value, location)
    if set(row) != {"candidate_key", "role", "text", "admissible"}:
        raise ValueError(f"{location} has missing or unknown fields")
    admissible = _mapping(row["admissible"], f"{location}.admissible")
    if set(admissible) != set(CONDITIONS):
        raise ValueError(f"{location}.admissible must contain allow and block")
    return CounterfactualCandidate(
        candidate_key=_string(row["candidate_key"], f"{location}.candidate_key"),
        role=_string(row["role"], f"{location}.role"),
        text=_string(row["text"], f"{location}.text"),
        admissible_allow=_boolean(admissible["allow"], f"{location}.admissible.allow"),
        admissible_block=_boolean(admissible["block"], f"{location}.admissible.block"),
    )


def _scenario_from_mapping(value: object, location: str) -> CounterfactualScenario:
    row = _mapping(value, location)
    expected = {"scenario_id", "axis", "source_basis", "candidates", "query_pairs"}
    if set(row) != expected:
        raise ValueError(f"{location} has missing or unknown fields")
    candidates = tuple(
        _candidate_from_mapping(candidate, f"{location}.candidates[{index}]")
        for index, candidate in enumerate(_sequence(row["candidates"], f"{location}.candidates"))
    )
    pairs = []
    for index, value in enumerate(_sequence(row["query_pairs"], f"{location}.query_pairs")):
        pair = _mapping(value, f"{location}.query_pairs[{index}]")
        if set(pair) != {"pair_id", "allow", "block"}:
            raise ValueError(f"{location}.query_pairs[{index}] has missing or unknown fields")
        pairs.append(
            QueryPair(
                pair_id=_string(pair["pair_id"], f"{location}.query_pairs[{index}].pair_id"),
                allow_query=_string(pair["allow"], f"{location}.query_pairs[{index}].allow"),
                block_query=_string(pair["block"], f"{location}.query_pairs[{index}].block"),
            )
        )
    return CounterfactualScenario(
        scenario_id=_string(row["scenario_id"], f"{location}.scenario_id"),
        axis=_string(row["axis"], f"{location}.axis"),
        source_basis=_string(row["source_basis"], f"{location}.source_basis"),
        candidates=candidates,
        query_pairs=tuple(pairs),
    )


def load_scenarios(path: Path) -> tuple[CounterfactualScenario, ...]:
    """Load and strictly validate the tracked controlled scenario bundle."""
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "dataset")
    if set(root) != {"schema_version", "dataset_id", "construction", "scenarios"}:
        raise ValueError("dataset has missing or unknown fields")
    if root["schema_version"] != 1:
        raise ValueError("counterfactual dataset schema version drifted")
    if root["dataset_id"] != "counterfactual-admissibility-pairs-v1":
        raise ValueError("counterfactual dataset identity drifted")
    if root["construction"] != "controlled_public_example_derived_and_synthetic":
        raise ValueError("counterfactual dataset construction drifted")
    scenarios = tuple(
        _scenario_from_mapping(value, f"dataset.scenarios[{index}]")
        for index, value in enumerate(_sequence(root["scenarios"], "dataset.scenarios"))
    )
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    pair_ids = [pair.pair_id for scenario in scenarios for pair in scenario.query_pairs]
    if not scenarios or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be nonempty and globally unique")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("query pair IDs must be globally unique")
    return scenarios


def expand_pairs(scenarios: Sequence[CounterfactualScenario]) -> tuple[CounterfactualPair, ...]:
    """Expand scenario-level candidate pools into pair-level immutable records."""
    return tuple(
        CounterfactualPair(
            pair_id=query_pair.pair_id,
            scenario_id=scenario.scenario_id,
            axis=scenario.axis,
            source_basis=scenario.source_basis,
            allow_query=query_pair.allow_query,
            block_query=query_pair.block_query,
            candidates=scenario.candidates,
        )
        for scenario in scenarios
        for query_pair in scenario.query_pairs
    )


_TOKEN = re.compile(r"[a-z0-9]+")


def query_token_jaccard(left: str, right: str) -> float:
    """Return case-folded alphanumeric token-set Jaccard similarity."""
    left_tokens = set(_TOKEN.findall(left.casefold()))
    right_tokens = set(_TOKEN.findall(right.casefold()))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def query_sequence_similarity(left: str, right: str) -> float:
    """Return a deterministic normalized character-sequence similarity."""
    return SequenceMatcher(None, left.casefold(), right.casefold(), autojunk=False).ratio()


def validate_dataset_contract(
    scenarios: Sequence[CounterfactualScenario],
    *,
    axes: Sequence[str],
    scenarios_per_axis: int,
    query_pairs_per_scenario: int,
    candidates_per_pair: int,
    minimum_query_token_jaccard: float,
) -> dict[str, object]:
    """Validate balance, fixed pools, and minimum counterfactual lexical overlap."""
    if not 0 <= minimum_query_token_jaccard <= 1:
        raise ValueError("minimum query token Jaccard must be in [0, 1]")
    expected_axes = tuple(axes)
    if len(set(expected_axes)) != len(expected_axes):
        raise ValueError("axes must be unique")
    counts = {axis: 0 for axis in expected_axes}
    similarities = []
    for scenario in scenarios:
        if scenario.axis not in counts:
            raise ValueError(f"unexpected axis {scenario.axis!r}")
        counts[scenario.axis] += 1
        if len(scenario.query_pairs) != query_pairs_per_scenario:
            raise ValueError("scenario query-pair count drifted")
        if len(scenario.candidates) != candidates_per_pair:
            raise ValueError("scenario candidate count drifted")
        for pair in scenario.query_pairs:
            similarity = query_token_jaccard(pair.allow_query, pair.block_query)
            if similarity < minimum_query_token_jaccard:
                raise ValueError(
                    f"pair {pair.pair_id!r} token Jaccard {similarity:.4f} is below minimum"
                )
            similarities.append(similarity)
    if any(count != scenarios_per_axis for count in counts.values()):
        raise ValueError("scenario count by axis drifted")
    pair_count = sum(len(scenario.query_pairs) for scenario in scenarios)
    return {
        "scenario_count": len(scenarios),
        "pair_count": pair_count,
        "case_count": pair_count * len(CONDITIONS),
        "candidate_judgment_count": pair_count * len(CONDITIONS) * candidates_per_pair,
        "axes": counts,
        "minimum_query_token_jaccard": min(similarities),
        "mean_query_token_jaccard": sum(similarities) / len(similarities),
    }


def prompt_payload(pair: CounterfactualPair, *, condition: str) -> str:
    """Serialize model-visible fields while excluding scorer labels and identities."""
    visible = {
        "query": pair.query(condition),
        "candidates": [
            {"candidate_key": candidate.candidate_key, "text": candidate.text}
            for candidate in pair.candidates
        ],
    }
    return json.dumps(visible, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def response_json_schema(candidate_count: int) -> dict[str, object]:
    """Return the exact provider-neutral response schema for one query condition."""
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
    ):
        raise ValueError("candidate count must be a positive integer")
    probabilities = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            label: {"type": "number", "minimum": 0, "maximum": 1} for label in ADMISSIBILITY_LABELS
        },
        "required": list(ADMISSIBILITY_LABELS),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_key": {"type": "string"},
                        "admissibility": probabilities,
                    },
                    "required": ["candidate_key", "admissibility"],
                },
            }
        },
        "required": ["candidates"],
    }


@dataclass(frozen=True)
class CandidatePrediction:
    candidate_key: str
    probabilities: ProbabilityVector


@dataclass(frozen=True)
class CounterfactualPrediction:
    case_id: str
    pair_id: str
    condition: str
    candidates: tuple[CandidatePrediction, ...]

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unexpected condition {self.condition!r}")
        if self.case_id != f"{self.pair_id}::{self.condition}":
            raise ValueError("prediction case identity is inconsistent")


def prediction_from_mapping(
    value: object,
    pair: CounterfactualPair,
    *,
    condition: str,
    tolerance: float = 0.0001,
) -> CounterfactualPrediction:
    """Parse a strict response and bind candidate count, keys, and order to the request."""
    if condition not in CONDITIONS:
        raise ValueError(f"unexpected condition {condition!r}")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("probability tolerance must be finite and nonnegative")
    row = _mapping(value, "prediction")
    if set(row) != {"candidates"}:
        raise ValueError("prediction has missing or unknown fields")
    raw_candidates = _sequence(row["candidates"], "prediction.candidates")
    if len(raw_candidates) != len(pair.candidates):
        raise ValueError("prediction candidate count differs from request")
    candidates = []
    for index, (raw_prediction, expected) in enumerate(
        zip(raw_candidates, pair.candidates, strict=True)
    ):
        candidate = _mapping(raw_prediction, f"prediction.candidates[{index}]")
        if set(candidate) != {"candidate_key", "admissibility"}:
            raise ValueError(f"prediction.candidates[{index}] has missing or unknown fields")
        key = _string(candidate["candidate_key"], f"prediction.candidates[{index}].candidate_key")
        if key != expected.candidate_key:
            raise ValueError("prediction candidate keys must reproduce request order")
        candidates.append(
            CandidatePrediction(
                candidate_key=key,
                probabilities=ProbabilityVector.from_mapping(
                    candidate["admissibility"],
                    labels=ADMISSIBILITY_LABELS,
                    tolerance=tolerance,
                    location=f"prediction.candidates[{index}].admissibility",
                ),
            )
        )
    return CounterfactualPrediction(
        case_id=pair.case_id(condition),
        pair_id=pair.pair_id,
        condition=condition,
        candidates=tuple(candidates),
    )


def _gold_prediction(pair: CounterfactualPair, condition: str) -> CounterfactualPrediction:
    candidates = []
    for candidate in pair.candidates:
        values = (1.0, 0.0, 0.0) if candidate.gold(condition) else (0.0, 1.0, 0.0)
        candidates.append(
            CandidatePrediction(
                candidate_key=candidate.candidate_key,
                probabilities=ProbabilityVector(labels=ADMISSIBILITY_LABELS, values=values),
            )
        )
    return CounterfactualPrediction(
        case_id=pair.case_id(condition),
        pair_id=pair.pair_id,
        condition=condition,
        candidates=tuple(candidates),
    )


def oracle_predictions(
    pairs: Sequence[CounterfactualPair],
) -> dict[str, CounterfactualPrediction]:
    """Return deterministic released-label oracle predictions."""
    return {
        pair.case_id(condition): _gold_prediction(pair, condition)
        for pair in pairs
        for condition in CONDITIONS
    }


def no_verifier_predictions(
    pairs: Sequence[CounterfactualPair],
) -> dict[str, CounterfactualPrediction]:
    """Return the query-blind keep-all baseline used by semantic retrieval alone."""
    predictions = {}
    for pair in pairs:
        for condition in CONDITIONS:
            predictions[pair.case_id(condition)] = CounterfactualPrediction(
                case_id=pair.case_id(condition),
                pair_id=pair.pair_id,
                condition=condition,
                candidates=tuple(
                    CandidatePrediction(
                        candidate_key=candidate.candidate_key,
                        probabilities=ProbabilityVector(
                            labels=ADMISSIBILITY_LABELS,
                            values=(1.0, 0.0, 0.0),
                        ),
                    )
                    for candidate in pair.candidates
                ),
            )
    return predictions


def _prediction_by_key(
    pair: CounterfactualPair,
    prediction: CounterfactualPrediction,
) -> dict[str, CandidatePrediction]:
    expected = tuple(candidate.candidate_key for candidate in pair.candidates)
    observed = tuple(candidate.candidate_key for candidate in prediction.candidates)
    if observed != expected:
        raise ValueError("prediction candidates differ from pair request order")
    return {candidate.candidate_key: candidate for candidate in prediction.candidates}


def _correct(prediction: CandidatePrediction, gold: bool) -> bool:
    expected = "admissible" if gold else "inadmissible"
    return prediction.probabilities.argmax() == expected


def _multiclass_brier(prediction: CandidatePrediction, gold: bool) -> float:
    expected = "admissible" if gold else "inadmissible"
    return sum(
        (probability - int(label == expected)) ** 2
        for label, probability in zip(
            prediction.probabilities.labels,
            prediction.probabilities.values,
            strict=True,
        )
    )


def pair_score_rows(
    pairs: Sequence[CounterfactualPair],
    predictions: Mapping[str, CounterfactualPrediction],
    *,
    model: str,
) -> tuple[dict[str, object], ...]:
    """Score paired directional behavior and query-invariant stable controls."""
    rows = []
    for pair in pairs:
        allow = predictions.get(pair.case_id("allow"))
        block = predictions.get(pair.case_id("block"))
        if allow is None or block is None:
            raise ValueError(f"missing prediction for pair {pair.pair_id!r}")
        allow_by_key = _prediction_by_key(pair, allow)
        block_by_key = _prediction_by_key(pair, block)
        focal = next(candidate for candidate in pair.candidates if candidate.role == "focal")
        focal_allow = allow_by_key[focal.candidate_key]
        focal_block = block_by_key[focal.candidate_key]
        focal_allow_score = focal_allow.probabilities.probability(
            "admissible"
        ) - focal_allow.probabilities.probability("inadmissible")
        focal_block_score = focal_block.probabilities.probability(
            "admissible"
        ) - focal_block.probabilities.probability("inadmissible")
        directional_margin = (focal_allow_score - focal_block_score) / 2

        candidate_correct = 0
        candidate_brier = 0.0
        candidate_unknown_argmax = 0
        candidate_unknown_probability = 0.0
        stable_correct = 0
        stable_judgments = 0
        stable_overflip = 0
        stable_candidates = 0
        for candidate in pair.candidates:
            condition_predictions = {
                "allow": allow_by_key[candidate.candidate_key],
                "block": block_by_key[candidate.candidate_key],
            }
            for condition, prediction in condition_predictions.items():
                gold = candidate.gold(condition)
                is_correct = _correct(prediction, gold)
                candidate_correct += int(is_correct)
                candidate_brier += _multiclass_brier(prediction, gold)
                candidate_unknown_argmax += int(prediction.probabilities.argmax() == "unknown")
                candidate_unknown_probability += prediction.probabilities.probability("unknown")
                if candidate.role != "focal":
                    stable_correct += int(is_correct)
                    stable_judgments += 1
            if candidate.role != "focal":
                stable_candidates += 1
                stable_overflip += int(
                    condition_predictions["allow"].probabilities.argmax()
                    != condition_predictions["block"].probabilities.argmax()
                )

        focal_allow_correct = _correct(focal_allow, True)
        focal_block_correct = _correct(focal_block, False)
        candidate_total = len(pair.candidates) * len(CONDITIONS)
        rows.append(
            {
                "model": model,
                "pair_id": pair.pair_id,
                "scenario_id": pair.scenario_id,
                "axis": pair.axis,
                "source_basis": pair.source_basis,
                "query_token_jaccard": query_token_jaccard(
                    pair.allow_query,
                    pair.block_query,
                ),
                "query_sequence_similarity": query_sequence_similarity(
                    pair.allow_query,
                    pair.block_query,
                ),
                "focal_allow_correct": focal_allow_correct,
                "focal_block_correct": focal_block_correct,
                "strict_focal_pair_consistency": int(focal_allow_correct and focal_block_correct),
                "focal_direction_accuracy": int(directional_margin > 0),
                "focal_directional_margin": directional_margin,
                "focal_unknown_argmax_rate": (
                    int(focal_allow.probabilities.argmax() == "unknown")
                    + int(focal_block.probabilities.argmax() == "unknown")
                )
                / 2,
                "stable_control_accuracy": stable_correct / stable_judgments,
                "stable_control_overflip_rate": stable_overflip / stable_candidates,
                "stable_control_correct_count": stable_correct,
                "stable_control_judgment_count": stable_judgments,
                "stable_control_overflip_count": stable_overflip,
                "stable_control_candidate_count": stable_candidates,
                "candidate_accuracy": candidate_correct / candidate_total,
                "candidate_correct_count": candidate_correct,
                "candidate_judgment_count": candidate_total,
                "candidate_multiclass_brier": candidate_brier / candidate_total,
                "candidate_multiclass_brier_sum": candidate_brier,
                "candidate_unknown_argmax_rate": candidate_unknown_argmax / candidate_total,
                "candidate_unknown_argmax_count": candidate_unknown_argmax,
                "candidate_unknown_probability": candidate_unknown_probability / candidate_total,
                "candidate_unknown_probability_sum": candidate_unknown_probability,
            }
        )
    return tuple(rows)


def aggregate_pair_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    """Aggregate pair-level metrics without treating paraphrases as bootstrap units."""
    if not rows:
        raise ValueError("pair metrics require nonempty rows")
    count = len(rows)

    def mean(field: str) -> float:
        return sum(_number(row[field], field) for row in rows) / count

    stable_correct = sum(
        int(_number(row["stable_control_correct_count"], "stable correct")) for row in rows
    )
    stable_total = sum(
        int(_number(row["stable_control_judgment_count"], "stable total")) for row in rows
    )
    stable_flips = sum(
        int(_number(row["stable_control_overflip_count"], "stable flips")) for row in rows
    )
    stable_candidates = sum(
        int(_number(row["stable_control_candidate_count"], "stable candidates")) for row in rows
    )
    candidate_correct = sum(
        int(_number(row["candidate_correct_count"], "candidate correct")) for row in rows
    )
    candidate_total = sum(
        int(_number(row["candidate_judgment_count"], "candidate total")) for row in rows
    )
    brier_sum = sum(
        _number(row["candidate_multiclass_brier_sum"], "candidate brier") for row in rows
    )
    unknown_count = sum(
        int(_number(row["candidate_unknown_argmax_count"], "candidate unknown")) for row in rows
    )
    unknown_probability = sum(
        _number(row["candidate_unknown_probability_sum"], "candidate unknown probability")
        for row in rows
    )
    return {
        "pair_count": count,
        "scenario_count": len({_string(row["scenario_id"], "scenario_id") for row in rows}),
        "strict_focal_pair_consistency": mean("strict_focal_pair_consistency"),
        "focal_direction_accuracy": mean("focal_direction_accuracy"),
        "mean_focal_directional_margin": mean("focal_directional_margin"),
        "focal_unknown_argmax_rate": mean("focal_unknown_argmax_rate"),
        "stable_control_accuracy": stable_correct / stable_total,
        "stable_control_overflip_rate": stable_flips / stable_candidates,
        "candidate_accuracy": candidate_correct / candidate_total,
        "candidate_multiclass_brier": brier_sum / candidate_total,
        "candidate_unknown_argmax_rate": unknown_count / candidate_total,
        "candidate_mean_unknown_probability": unknown_probability / candidate_total,
        "mean_query_token_jaccard": mean("query_token_jaccard"),
        "mean_query_sequence_similarity": mean("query_sequence_similarity"),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def scenario_stratified_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap whole scenarios within axes and report an equal-axis macro mean."""
    if not rows or replicates < 1:
        raise ValueError("bootstrap requires rows and positive replicates")
    by_axis_scenario: dict[str, dict[str, list[Mapping[str, object]]]] = {}
    for row in rows:
        axis = _string(row["axis"], "axis")
        scenario = _string(row["scenario_id"], "scenario_id")
        by_axis_scenario.setdefault(axis, {}).setdefault(scenario, []).append(row)

    def scenario_mean(scenario_rows: Sequence[Mapping[str, object]]) -> float:
        return sum(_number(row[metric], metric) for row in scenario_rows) / len(scenario_rows)

    def equal_axis_mean(sampled: Mapping[str, Sequence[Sequence[Mapping[str, object]]]]) -> float:
        axis_means = []
        for axis in sorted(sampled):
            scenario_means = [scenario_mean(scenario_rows) for scenario_rows in sampled[axis]]
            axis_means.append(sum(scenario_means) / len(scenario_means))
        return sum(axis_means) / len(axis_means)

    observed = {axis: list(scenarios.values()) for axis, scenarios in by_axis_scenario.items()}
    estimate = equal_axis_mean(observed)
    generator = random.Random(seed)
    draws = []
    for _index in range(replicates):
        sampled = {}
        for axis, scenarios in sorted(by_axis_scenario.items()):
            units = list(scenarios.values())
            sampled[axis] = [units[generator.randrange(len(units))] for _unit in units]
        draws.append(equal_axis_mean(sampled))
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_lower": _quantile(draws, 0.025),
        "ci_upper": _quantile(draws, 0.975),
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "scenario",
        "bootstrap_stratification": "axis",
        "axis_count": len(by_axis_scenario),
        "scenario_count": sum(len(scenarios) for scenarios in by_axis_scenario.values()),
        "pair_count": len(rows),
    }
