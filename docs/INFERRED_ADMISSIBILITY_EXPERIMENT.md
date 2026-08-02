# Text-Inferred Admissibility Diagnostic

This dev-only diagnostic implements the remaining oracle-versus-inferred comparison
recommended by the external review. It does not replace the frozen natural evaluation
and does not turn released source fields into a deployable policy engine.

## Question

The natural evaluation shows what trusted namespace and released governance fields can
do. This experiment asks a different question: when a verifier sees only query text,
candidate text, and non-label time/order evidence, how much of that oracle headroom can
it recover without denying required evidence?

The four route arms are:

| Arm | Behavior |
| --- | --- |
| `namespace_dense` | Frozen namespace-local top-20 ranking, with no additional filtering |
| `released_oracle` | Removes candidates that released policy or lifecycle fields establish as incompatible |
| `text_inferred` | Removes candidates above a model-specific violation threshold selected on calibration only |
| `abstaining_verifier` | Applies the same filter only when model-reported unknown probability is below a selected cutoff |

The inferred arms never see source IDs, namespace IDs, embeddings, required-evidence
labels, released intent, released policy, lifecycle state, or distractor flags. The
prompt explicitly separates admissibility from relevance. An `unknown` prediction is
therefore an informative outcome when governance facts are absent from text.
Each candidate also receives a direct three-way admissibility probability. This avoids
treating the maximum of separate policy and lifecycle marginals as though it were a
calibrated probability for their logical union.

## Frozen Sample

The sample contains 96 public development queries, selected by a stable hash: 32 RHELM
current-state, 32 MemOps current-state, and 32 MemOps history queries. Within each
stratum, eight queries are calibration-only and 24 are analysis-only. Every query uses
the first 20 candidates from the previously frozen namespace-dense ranking; candidates
are not reranked or injected. Query and candidate strings use the same deterministic
2,048-byte UTF-8 prefix for every provider. This prevents a verifier from receiving
unbounded source text when the frozen retriever used a 512-token encoder limit.

This balanced diagnostic sample is not a population estimate. RHELM contributes a
clean false-deny control, while MemOps supplies released distractor and lifecycle
cases. Results must be reported by stratum as well as in the equal-stratum macro.

## Selection and Evaluation

For each model, calibration chooses a violation threshold that maximizes known
violation recall subject to at most one percent false denial of required anchors.
The abstaining arm additionally selects an unknown-probability cutoff. If no grid
setting satisfies the constraint, the arm retains every candidate. Analysis queries
are never used for threshold selection.

Classification reporting includes class prevalence, accuracy, balanced accuracy,
Brier score, ECE, ROC-AUC, PR-AUC, violation precision/recall, abstention coverage, and
required-anchor false-deny rate. Downstream reporting includes evidence recall,
matched-recall feasibility, conditional and penalized admissibility risk, route width,
and the fraction of the released-oracle gap closed. Paired route deltas use 10,000
namespace-group bootstrap replicates within each source/intent stratum. Classification
and filter summaries are also reported per stratum and as a true equal-stratum macro.

## Models and Boundaries

The exact provider/model bindings are frozen in
`experiments/inferred_admissibility_protocol.json`. The current cross-provider panel is
GPT-5.6 Sol, DeepSeek-V4-Pro, Gemini 3.6 Flash, and Claude Sonnet 5. All receive the
same semantic prompt and output contract. Provider-specific request encodings are
adapters, not prompt variants. When a provider does not support a syntactic JSON Schema
constraint, its adapter removes only that constraint and the client still validates the
response against the original candidate-count, order, and probability contract.

Before the development sample is sent, a 20-candidate synthetic fixture exercises
each provider adapter. Fixture receipts are bound to the protocol, prompt, fixture
payload, and adapter hashes; they contain no model output. Full execution checkpoints
are additionally bound to the exact protocol and per-case visible payload so stale
responses cannot be silently reused after input drift. Provider usage reporting keeps
token counts, paid cost, call counts, and latency percentiles alongside quality.
The execution command refuses to run a provider without its current fixture receipt.

Raw public-source text and provider responses remain local and ignored by Git. Only
content-free aggregate tables, model IDs, usage, hashes, and failure counts may be
published. There is no answer generation, reader, judge, held-out access, evaluation
retuning, or official benchmark claim.
