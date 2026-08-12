# Natural Support Controls

This zero-call post-hoc analysis uses the frozen natural-corpus embeddings and exact
dense ranker. It changes no embedding, label, query, or tuned setting. The primary
route limit is 20 and the primary matched-recall target is 0.8.

## Main comparison

| Arm | Recall | Feasible | Risk | Residual | Any violation | Mean violations | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global dense | 0.4316 | 0.2374 | 0.7981 | 0.7789 | 0.5285 | 2.080 | 90121.7 |
| Size-matched random partition | 0.0610 | 0.0242 | 0.9780 | 0.9758 | 0.1757 | 0.497 | 1540.1 |
| Namespace pre-filter | 0.5329 | 0.3113 | 0.7127 | 0.7127 | 0.3964 | 1.393 | 1551.2 |
| Global top-20 then namespace filter | 0.4316 | 0.2374 | 0.7784 | 0.7784 | 0.3721 | 1.165 | 90121.7 |

The random-partition arm preserves the source-level namespace label counts but breaks
their provenance assignment. It tests whether candidate-pool size alone explains the
namespace result. Post-filter arms test whether a shallow global retrieval can recover
the same support as filtering before ranking. Residual risk removes scope violations
from the numerator and retains released policy/lifecycle violations and unresolved
labels, so it is not mechanically reduced by the namespace check itself.
The two known-violation columns are calculated only on feasible matched-recall
prefixes. Unlike fractional risk, neither the any-violation indicator nor the count
can be reduced by inserting additional established-positive records before the
recall target.

Sensitivity files report target recall 0.5--1.0 and infeasibility cost 0--1. The
few-cluster file reports leave-one-namespace-out ranges and an exact two-sided sign-flip
test for the seven RHELM groups. This is a frozen diagnostic, not a new benchmark run.
It makes zero provider, reader, judge, or paid calls and publishes no raw text, IDs,
embeddings, prompts, responses, or private payloads.
