# Natural cross-judge audit

## Result

The blinded `gpt-5.1-2025-11-13` audit agreed with the frozen Claude Haiku 4.5 labels on
**0.865** of 200 exact-deduplicated outputs
(item-bootstrap 95% CI **[0.815,
0.910]**). Cohen's kappa was
**0.728** and Gwet's AC1 was
**0.732**. The pre-existing continuation threshold of
0.85 was **met**.

The original DeepSeek--Gemini reader panel agreed on
**0.881** of 134 sampled outputs; the later
sequential GPT-reader subgroup agreed on **0.833**
of 66. These subgroup estimates are diagnostic and their intervals are wide.

For the 0--10 answer-quality score, exact agreement was
**0.240**, agreement within one point was
**0.755**, and mean absolute error was
**1.245**.

## Scope

The original GPT-5 Pro attempt accepted zero benchmark labels before a 512-token completion-ceiling failure; the frozen recovery migrated to GPT-5.1 before outcome inspection.

This is a post-hoc but outcome-independent robustness audit, not an independently
preregistered replication. The 200 payloads were selected without inspecting reader
answers or either judge's labels, using near-equal quotas over three readers, two
sources, and the three common primary routes. Exact duplicate judge payloads were
called once. Reader estimates remain separate and no model pooling is introduced.

The primary endpoint is semantic answer-correctness agreement. Protected- and
stale-disclosure agreement is secondary because many sampled payloads are negative or
not disclosure-evaluable. No answer, reference answer, prompt, or judge rationale is
included in the public artifacts.
