# Claude Opus 5 Paired-Exposure Replication

This is a separately executed, controlled reader replication. It reuses the frozen
16 scenarios, 192 paired units, and exposed/withheld construction from the original
three-reader experiment. It does not alter or pool the GPT-5.6 Sol, Gemini 3.6 Flash,
or DeepSeek V4 Pro estimates, and it is not an official benchmark result.

| Reader | Relevant + admissible ATE | Relevant + inadmissible ATE | Selectivity gap (95% CI) |
| --- | ---: | ---: | ---: |
| `claude-opus-5` | 0.9688 | 0.1250 | 0.8438 [0.6875, 0.9688] |

The relevant-inadmissible exposure effect is
0.1250
[0.0000, 0.2812].
The selectivity-gap interval excludes zero, replicating the qualitative prompt-level
pattern with a fourth provider. The nonzero inadmissible point estimate also preserves
the paper's central boundary: reader restraint is imperfect after inadmissible content
has entered the prompt.

The execution used exactly `claude-opus-5`, thinking disabled, medium effort, one
accepted response per request, no transport retry, no output repair, no selective
rerun, and no judge. The content-free publication contains derived pair scores,
aggregate metrics, uncertainty, cost/latency summaries, and provenance hashes. It
contains no prompts, candidate text, literal target markers, provider responses,
answer text, credentials, or private data.
