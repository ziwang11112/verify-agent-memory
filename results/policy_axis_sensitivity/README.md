# Policy-Axis Sensitivity

This frozen, zero-call sensitivity reuses the published global-dense and
namespace-dense top-20 routes. It omits only the released policy-disallowed
predicate from admissibility scoring; source-required anchors, scope, lifecycle,
rankings, route limits, and the 0.8 recall target remain unchanged.

| Axis family | Arm | Recall | Feasible | Penalized upper risk | Known risk | Coverage | Bounds | Any violation | Mean count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scope + policy + lifecycle | global dense | 0.4316 | 0.2374 | 0.7981 | 0.1893 | 0.9918 | [0.1873, 0.1955] | 0.5285 | 2.080 |
| Scope + policy + lifecycle | namespace dense | 0.5329 | 0.3113 | 0.7127 | 0.1198 | 0.9920 | [0.1181, 0.1261] | 0.3964 | 1.393 |
| Scope + lifecycle (policy omitted) | global dense | 0.4316 | 0.2374 | 0.7868 | 0.1031 | 0.9918 | [0.1025, 0.1106] | 0.3564 | 1.080 |
| Scope + lifecycle (policy omitted) | namespace dense | 0.5329 | 0.3113 | 0.6935 | 0.0176 | 0.9920 | [0.0173, 0.0253] | 0.1343 | 0.181 |

The direction of the namespace comparison is evaluated in `paired_deltas.csv`
with a paired 10,000-sample namespace-group bootstrap within each source.
No query IDs, memory IDs, source text, embeddings, prompts, or responses are
written by this analysis.
