"""Materialize private route-to-reader cases from the pinned public-source cache."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verify_agent_memory.natural_end_to_end import (  # noqa: E402
    NaturalCandidate,
    NaturalEndToEndCase,
    truncate_utf8,
)

DEFAULT_LEGACY_ROOT = ROOT.parent / "bomi-codex-starter"
DEFAULT_OUTPUT = ROOT / "tmp" / "natural_end_to_end" / "cases.jsonl.gz"
DEFAULT_MANIFEST = ROOT / "tmp" / "natural_end_to_end" / "materialization_manifest.json"
GLOBAL_TASK = "flat_dense/25ea22a7b2be"
NAMESPACE_TASK = "namespace_dense/ed4a7e3ae49d"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_gzip_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            for row in rows:
                compressed.write(_canonical_bytes(row) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_api(legacy_root: Path) -> dict[str, object]:
    if not (legacy_root / "bomi" / "bench" / "stage3_natural_corpus_admissibility.py").is_file():
        raise FileNotFoundError(f"legacy natural-corpus runtime not found: {legacy_root}")
    sys.path.insert(0, str(legacy_root))
    from bomi.bench.stage3_admissibility_audit import load_rhelm_qa_index
    from bomi.bench.stage3_natural_route_store import load_route_bundle
    from bomi.bench.stage3_rhelm_evidence_resolution import (
        iter_rhelm_question_resolutions,
    )

    from scripts.run_stage3_natural_corpus_embedding_pilot import SOURCE_CACHE
    from scripts.run_stage3_natural_corpus_public_eval import (
        ROUTE_BUNDLES,
        ROUTE_CHECKPOINT_PATH,
        _load_eval_inputs,
    )

    return {
        "SOURCE_CACHE": SOURCE_CACHE,
        "ROUTE_BUNDLES": ROUTE_BUNDLES,
        "ROUTE_CHECKPOINT_PATH": ROUTE_CHECKPOINT_PATH,
        "iter_rhelm_question_resolutions": iter_rhelm_question_resolutions,
        "load_eval_inputs": _load_eval_inputs,
        "load_rhelm_qa_index": load_rhelm_qa_index,
        "load_route_bundle": load_route_bundle,
    }


def _memory_key(source: str, memory_id: str) -> str:
    return hashlib.sha256(f"{source}|{memory_id}".encode()).hexdigest()[:24]


def _visible_order(memory: object) -> str | None:
    source = str(memory.source)  # type: ignore[attr-defined]
    chronology = str(memory.chronology)  # type: ignore[attr-defined]
    if source == "rhelm":
        return chronology
    return chronology.split("/", maxsplit=1)[-1]


def _lifecycle_compatible(query: object, memory: object) -> bool | None:
    state = str(memory.state)  # type: ignore[attr-defined]
    if state == "uncertain":
        return None
    return not (
        str(query.query_intent) == "current_state"  # type: ignore[attr-defined]
        and state in {"stale", "superseded"}
    )


def _candidate(
    *,
    source: str,
    query: object,
    memory: object,
    rank: int,
    text: str,
    maximum_bytes: int,
) -> NaturalCandidate:
    return NaturalCandidate(
        memory_key=_memory_key(source, str(memory.memory_id)),  # type: ignore[attr-defined]
        rank=rank,
        text=truncate_utf8(text, maximum_bytes),
        visible_order=_visible_order(memory),
        required_evidence=str(memory.memory_id)  # type: ignore[attr-defined]
        in set(query.required_memory_ids),  # type: ignore[attr-defined]
        scope_allowed=str(memory.namespace_id)  # type: ignore[attr-defined]
        == str(query.namespace_id),  # type: ignore[attr-defined]
        policy_allowed=not bool(memory.released_distractor),  # type: ignore[attr-defined]
        lifecycle_compatible=_lifecycle_compatible(query, memory),
    )


def _rhelm_gold(api: Mapping[str, object], eval_query_ids: set[str]) -> dict[str, dict[str, Any]]:
    source_cache = Path(api["SOURCE_CACHE"])
    data_root = source_cache / "RHELM" / "data"
    qa_index = api["load_rhelm_qa_index"](data_root)  # type: ignore[operator]
    rows = {}
    for resolution in api["iter_rhelm_question_resolutions"](data_root):  # type: ignore[operator]
        if not (resolution.conversation_only and resolution.all_references_unique):
            continue
        query_id = f"{resolution.namespace_id}/{resolution.query_id}"
        if query_id in eval_query_ids:
            rows[query_id] = dict(qa_index[(resolution.namespace_id, resolution.query_id)])
    if set(rows) != eval_query_ids:
        raise RuntimeError("RHELM answer gold does not cover the evaluation queries")
    return rows


def _memops_gold(api: Mapping[str, object], eval_query_ids: set[str]) -> dict[str, dict[str, Any]]:
    source_cache = Path(api["SOURCE_CACHE"])
    injected_root = (
        source_cache / "MemOps" / "generated_result" / "4-inject_evidence_with_distractors"
    )
    rows = {}
    for path in sorted(injected_root.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        for raw_answer in case["answer"]:
            answer = dict(raw_answer)
            query_id = f"{path.stem}/{answer['question_pair_id']}/{answer['evaluation_setting']}"
            if query_id in eval_query_ids:
                answer["operation_type"] = case["operation_type"]
                rows[query_id] = answer
    if set(rows) != eval_query_ids:
        raise RuntimeError("MemOps answer gold does not cover the evaluation queries")
    return rows


def _answer_fields(
    source: str,
    row: Mapping[str, Any],
) -> tuple[str, dict[str, object], tuple[str, ...]]:
    if source == "rhelm":
        metadata = {
            key: row[key] for key in ("question_type", "characteristics", "options") if key in row
        }
        return str(row["answer"]), metadata, ()
    rubric = dict(row.get("judge_rubric") or {})
    protected = tuple(
        dict.fromkeys(
            str(value).strip() for value in rubric.get("must_not_include", []) if str(value).strip()
        )
    )
    metadata = {
        key: row[key]
        for key in (
            "evaluation_type",
            "evaluation_category",
            "evaluation_setting",
            "operation_type",
            "gold_memory_state",
            "judge_rubric",
            "diagnostic_checks",
            "difficulty",
        )
        if key in row
    }
    return str(row["expected_answer"]), metadata, protected


def _materialized_candidates(
    *,
    source: str,
    route_name: str,
    observation: object,
    query: object,
    memories: Mapping[str, object],
    runtime_text: Mapping[tuple[str, str], str],
    candidate_depth: int,
    reader_memory_max_bytes: int,
    memory_keys: dict[str, str],
    counters: Counter[str],
) -> tuple[NaturalCandidate, ...]:
    result = []
    for rank, memory_id in enumerate(
        observation.ranked_memory_ids[:candidate_depth],  # type: ignore[attr-defined]
        start=1,
    ):
        memory = memories.get(memory_id)
        if memory is None:
            raise RuntimeError(f"route memory is absent: {source}/{memory_id}")
        text = runtime_text[("memory", memory.text_sha256)]  # type: ignore[attr-defined]
        candidate = _candidate(
            source=source,
            query=query,
            memory=memory,
            rank=rank,
            text=text,
            maximum_bytes=reader_memory_max_bytes,
        )
        previous = memory_keys.setdefault(candidate.memory_key, memory_id)
        if previous != memory_id:
            raise RuntimeError("memory-key digest collision")
        result.append(candidate)
        counters[f"{source}_{route_name}_candidate_rows"] += 1
        counters[f"{source}_{route_name}_truncated_rows"] += int(
            len(text.encode("utf-8")) > reader_memory_max_bytes
        )
    return tuple(result)


def materialize(
    *,
    legacy_root: Path,
    output: Path,
    manifest_path: Path,
    candidate_depth: int,
    reader_memory_max_bytes: int,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build the exact private case bundle and a content-free integrity manifest."""
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError("refusing to overwrite an existing materialization")
    api = _legacy_api(legacy_root.resolve())
    inputs = api["load_eval_inputs"](include_texts=True, include_model=False)  # type: ignore[operator]
    runtime_text = {row.item.embedding_identity: row.prepared_text for row in inputs.runtime_rows}
    checkpoint_path = Path(api["ROUTE_CHECKPOINT_PATH"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    records = {record["task_id"]: record for record in checkpoint["records"]}
    route_root = Path(api["ROUTE_BUNDLES"])
    route_hashes: dict[str, str] = {}
    cases = []
    counters: Counter[str] = Counter()
    memory_keys: dict[str, str] = {}

    for source in ("rhelm", "memops"):
        corpus = inputs.corpora[source]
        queries = {query.query_id: query for query in corpus.queries if query.split == "eval"}
        memories = {
            memory.memory_id: memory for memory in corpus.memories if memory.split == "eval"
        }
        if source == "rhelm":
            gold = _rhelm_gold(api, set(queries))
        else:
            gold = _memops_gold(api, set(queries))

        routes = {}
        for route_name, task_suffix in (
            ("global", GLOBAL_TASK),
            ("namespace", NAMESPACE_TASK),
        ):
            record = records[f"{source}/{task_suffix}"]
            path = route_root / record["filename"]
            route_hashes[f"{source}/{route_name}"] = _sha256_file(path)
            observations = api["load_route_bundle"](path)  # type: ignore[operator]
            routes[route_name] = {observation.query_id: observation for observation in observations}
        if set(routes["global"]) != set(queries) or set(routes["namespace"]) != set(queries):
            raise RuntimeError("route query coverage differs from the frozen evaluation population")

        for query_id in sorted(queries):
            query = queries[query_id]
            query_text = runtime_text[("query", query.query_text_sha256)].split(
                "\nQuery:",
                maxsplit=1,
            )[-1]
            expected_answer, metadata, protected = _answer_fields(source, gold[query_id])

            case = NaturalEndToEndCase(
                case_id=hashlib.sha256(f"{source}|{query_id}".encode()).hexdigest()[:32],
                source=source,
                group_id=str(query.namespace_id),
                query_text=query_text,
                query_visible_time=query.cutoff_date,
                query_intent=str(query.query_intent),
                expected_answer=expected_answer,
                answer_metadata=metadata,
                protected_targets=protected,
                anchor_total=len(query.required_memory_ids),
                global_candidates=_materialized_candidates(
                    source=source,
                    route_name="global",
                    observation=routes["global"][query_id],
                    query=query,
                    memories=memories,
                    runtime_text=runtime_text,
                    candidate_depth=candidate_depth,
                    reader_memory_max_bytes=reader_memory_max_bytes,
                    memory_keys=memory_keys,
                    counters=counters,
                ),
                namespace_candidates=_materialized_candidates(
                    source=source,
                    route_name="namespace",
                    observation=routes["namespace"][query_id],
                    query=query,
                    memories=memories,
                    runtime_text=runtime_text,
                    candidate_depth=candidate_depth,
                    reader_memory_max_bytes=reader_memory_max_bytes,
                    memory_keys=memory_keys,
                    counters=counters,
                ),
            )
            cases.append(case)
            counters[f"{source}_queries"] += 1
            counters[f"{source}_protected_evaluable"] += int(case.protected_disclosure_evaluable)

    cases.sort(key=lambda case: case.case_id)
    if len(cases) != 3767 or len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("materialized case count or identity differs from the frozen contract")
    _write_gzip_jsonl(output, [case.as_dict() for case in cases])
    manifest = {
        "schema_version": 1,
        "protocol_id": "natural-heldout-route-to-reader-v1",
        "status": "private_materialization_from_public_sources",
        "legacy_runtime_commit": _git_head(legacy_root),
        "route_checkpoint_sha256": _sha256_file(checkpoint_path),
        "route_bundle_sha256": dict(sorted(route_hashes.items())),
        "candidate_depth": candidate_depth,
        "reader_memory_max_utf8_bytes": reader_memory_max_bytes,
        "case_count": len(cases),
        "counts": dict(sorted(counters.items())),
        "case_bundle_sha256": _sha256_file(output),
        "raw_text_tracked": False,
        "provider_calls": 0,
        "paid_calls": 0,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _git_head(root: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--reader-memory-max-bytes", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = materialize(
        legacy_root=args.legacy_root,
        output=args.output,
        manifest_path=args.manifest,
        candidate_depth=args.candidate_depth,
        reader_memory_max_bytes=args.reader_memory_max_bytes,
        overwrite=args.force,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
