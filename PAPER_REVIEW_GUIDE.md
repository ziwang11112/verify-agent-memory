# Paper Review Guide

This is the entry point for a collaborator reviewing or revising the workshop paper.
The source of truth is the current `agent/retrieval-experiment-code` branch and PR #2.
The paper is a verification and evaluation paper, not a new memory-index proposal.

## Target and Current State

- Venue: NeurIPS 2026 workshop, *Who Verifies the Agents? Toward Reliable Agent
  Development*.
- Working title: *The Wrong Memory at the Right Time: Metadata Reliability and
  Enforcement Boundaries in Long-Term Agent Memory*.
- Current build: 9 content pages, 22 pages including references, appendices, and the
  checklist.
- Evidence contract: C1-C12, all backed by normalized content-free artifacts.
- Main empirical chain:

```text
Candidate support -> Metadata reliability -> Selective verification
                  -> Prompt exposure -> Answer disclosure
```

## Central Claim

A semantically relevant memory can still be inadmissible for the current principal,
purpose, intent, or time. Reliable eligibility information should constrain support
before prompt assembly. Semantic ranking and reader restraint do not reliably repair
an incorrectly specified candidate set after the fact.

## Contributions We Can Defend

1. A query-conditioned, three-valued retrieval-admissibility target that separates
   relevance from scope, policy, and lifecycle compatibility.
2. An auditable evaluation protocol linking stored, retrieved, exposed, and disclosed
   records, including matched-recall risk, unresolved-label bounds, feasibility, and
   typed violations.
3. A mechanism-oriented empirical decomposition showing where trusted constraints
   help, how metadata corruption fails asymmetrically, why text-only verification
   over-filters, and what happens when inadmissible evidence reaches the reader.

The novelty is the common measurement and enforcement analysis across these stages.
It is not the individual use of namespaces, policy tags, lifecycle labels, dense
retrieval, or paired prompting in isolation.

## Experiment Map

| Experiment | Population | Systems | Primary question |
| --- | --- | --- | --- |
| Natural support | Public RHELM and MemOps; 87 evaluation groups, 182,908 memories, 3,767 queries | Nine frozen retrieval arms; global and namespace dense are primary | Does trusted candidate support improve utility and risk at fixed budgets? |
| Metadata reliability | Same frozen natural rankings; ten seeds per corruption point | Namespace, policy, lifecycle, and combined filters | Which metadata supplies value, and how do missing, false-deny, fail-open, and swap errors change it? |
| Natural verification | 96 public-development cases: 24 calibration, 72 analysis | Released-field oracle, GPT-5.6 Sol, Gemini 3.6 Flash | Can text-only verification recover released-field headroom without losing required evidence? |
| Controlled verification | 16 constructed scenarios; 32 focal and 64 stable pairs | Keep-all, released-label oracle, GPT-5.6 Sol, Gemini 3.6 Flash, DeepSeek V4 Pro | Do models follow focal rule changes while preserving stable evidence? |
| Paired exposure | Same 16 scenarios; 192 pairs and 384 scored requests per reader | Original three-reader execution plus separate Claude Opus 5 replication | Once a candidate is exposed, how does disclosure change by relevance and admissibility? |
| GateMem audit | Separate observational appendix populations | Two fixed OpenAI readers and frozen G0/G1 routes | How are natural exposure, disclosure, leakage, utility, and refusal associated? |

These populations and estimands are never pooled.

## Model Roles

- `Qwen/Qwen3-Embedding-8B`: frozen retrieval encoder only; 4,096-dimensional
  normalized vectors. It is not a reader or judge.
- `gpt-5.6-sol`: OpenAI Sol tier, reasoning `none`; not Luna.
- `gemini-3.6-flash`: Google Flash tier, thinking `minimal`; not Pro.
- `deepseek-v4-pro`: DeepSeek Pro tier, thinking disabled; not Flash.
- `claude-opus-5`: independently executed Anthropic reader replication, thinking
  disabled and medium effort. It is not pooled with the original panel.
- GPT-4o and GPT-4o mini occur only in the older observational GateMem appendix.

The controlled readers form a heterogeneous cross-provider robustness panel, not a
same-tier capability leaderboard.

## Headline Results

- At top-20, trusted namespace support changes recall from 0.432 to 0.533 and
  penalized admissibility upper risk from 0.798 to 0.713.
- At top-100, namespace support changes recall by +0.149, feasible rate by +0.223,
  and penalized admissibility upper risk by -0.274. The measured candidate work drops
  58.1-fold under exact search; this is not a wall-clock latency claim.
- The released-field oracle reduces natural-development admissibility risk by -0.032
  (95% CI [-0.065, -0.007]) without changing recall. The tested text-only filters do
  not preserve that utility-risk improvement.
- Controlled stable overflip is 0.203 for GPT-5.6 Sol, 0.297 for Gemini 3.6 Flash,
  and 0.266 for DeepSeek V4 Pro.
- Original paired-exposure selectivity gaps are 0.906, 0.812, and 0.656 for GPT-5.6
  Sol, Gemini 3.6 Flash, and DeepSeek V4 Pro. Every interval excludes zero.
- DeepSeek has a relevant-inadmissible exposure effect of 0.156
  (95% CI [0.031, 0.312]).
- The separate Claude Opus 5 replication has a selectivity gap of 0.844
  (95% CI [0.688, 0.969]) and a relevant-inadmissible point estimate of 0.125
  (95% CI [0.000, 0.281]).

Use generated macros in `paper/generated/paper_numbers.tex` rather than manually
copying empirical numbers into LaTeX.

## Claims We Must Not Make

- A new state-of-the-art memory index, router, or deployable governance system.
- An official GateMem, RHELM, MemOps, or other benchmark score.
- A production-safety guarantee or natural-prevalence estimate from constructed data.
- A deployable method that infers latent scope, policy, intent, or lifecycle state.
- A causal natural-route exposure effect; the GateMem analysis is observational.
- An isolated causal moderator effect of admissibility in the paired intervention.
- A pooled four-reader result or model capability ranking.
- Universal failure of clustering or thresholds beyond the frozen tested variants.

Do not reintroduce the historical Bayesian/CRP development path or the discarded
human-agreement experiment. Neither belongs to this paper's evidence story.

## Current Review Risks

1. The paper must make its evaluation contribution feel substantive without implying
   that admissibility, namespaces, or policy metadata are newly invented.
2. The constructed verification and exposure populations support mechanism claims,
   not natural prevalence or production safety.
3. The natural retrieval audit has no downstream reader; the controlled exposure
   study closes a different part of the chain on a different population.
4. Released metadata is an oracle-like evidence source. No latent-field inference or
   noisy production identity system is demonstrated.
5. The main text is exactly at the 9-page limit and should lose roughly half a page
   before submission.
6. Reader configurations differ by provider and reasoning control; comparisons are
   robustness checks, not a leaderboard.

## Editing Boundaries

- Edit prose in `paper/main.tex` and references in `paper/references.bib`.
- Read `CLAIM_CONTRACT.md`, `paper/CLAIM_MAP.md`, and
  `paper/FIGURE_CONTRACT.md` before changing empirical wording.
- Do not manually edit `paper/generated/` or tracked result CSV/JSON files.
- A changed empirical number requires updating normalized evidence and its claim
  contract, then regenerating artifacts. Pure prose edits do not.
- Preserve double-blind wording and the 4-9 content-page limit.
- Keep C8, the original three-reader execution, separate from C12, the Claude
  replication.

## Recommended Reading Order

1. `paper/main.tex`
2. `paper/FIGURE_CONTRACT.md`
3. `paper/CLAIM_MAP.md`
4. `CLAIM_CONTRACT.md`
5. `docs/EXPERIMENT_METHODS.md`
6. `evidence/README.md`
7. `paper/BUILD_REPORT.md`

## Requested Review Output

Return findings before rewritten prose:

1. P0 blockers that could cause rejection or invalidate a claim.
2. P1 story, novelty, experimental-design, and missing-related-work issues.
3. P2 clarity and compression edits.
4. A line-referenced patch for `paper/main.tex` and `paper/references.bib` only.
5. A list of any proposed empirical change that would require reopening the frozen
   evidence contract. Do not silently make such a change.

Focus especially on whether the title, abstract, contributions, result headings, and
conclusion tell the same story; whether the distinction between natural and
constructed evidence is unmistakable; and what can be cut to create page-limit
margin.

## Verification Commands

```powershell
uv sync --extra dev --extra paper
uv run --extra dev --extra paper python -m scripts.build_paper_artifacts
uv run --extra dev --extra paper python -m scripts.verify_paper
uv run --extra dev --extra paper python -m scripts.compile_paper
uv run --extra dev python -m pytest -q
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
```

For a flat Overleaf package with one self-contained `main.tex` and all figures under
`figure/`:

```powershell
uv run --extra dev python -m scripts.export_overleaf --output tmp/overleaf_export
```
