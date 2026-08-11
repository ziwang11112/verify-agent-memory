# Frozen Operating Curves and Few-Scenario Robustness

The verifier table sweeps a fixed 0.00--1.00 threshold grid on the frozen 72-case
analysis partition. The original calibration-selected threshold is marked but never
changed. Curves expose precision--recall, retained-candidate coverage, required-anchor
false denial, evidence recall, feasible rate, and matched-recall risk.
Candidate precision, recall, and false denial are analysis-candidate micro averages;
route outcomes are equal-stratum macro averages, matching the primary diagnostic.

## Dev-selected verifier points

| Reader | Threshold | Precision | Violation recall | False deny | Route risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.6-sol` | 0.95 | 0.3796 | 0.2364 | 0.0531 | 0.8121 |
| `gemini-3.6-flash` | 0.95 | 0.3793 | 0.0500 | 0.0177 | 0.7733 |
Gemini has no grid point at or below 1% anchor false denial.
OpenAI has 1 grid point(s) at or below 1% anchor false denial; the best violation recall there is 0.0000.

## Leave-one-scenario-out selectivity

| Reader | Mean gap | LOO range | Exact sign-flip p |
| --- | ---: | ---: | ---: |
| `deepseek-v4-pro` | 0.6562 | [0.6333, 0.7000] | 0.000244 |
| `gemini-3.6-flash` | 0.8125 | [0.8000, 0.8667] | 0.000061 |
| `gpt-5.6-sol` | 0.9062 | [0.9000, 0.9333] | 0.000031 |

These are post-hoc robustness analyses of frozen outputs, not new model runs or
official benchmark results. Providers remain separate. No response, prompt, raw
text, case ID, candidate ID, or scenario ID is published here.
