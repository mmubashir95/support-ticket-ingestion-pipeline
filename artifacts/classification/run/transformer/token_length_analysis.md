# DistilBERT Token-Length Analysis

Lengths use the matching tokenizer with special tokens, without padding or truncation.

| Statistic | Tokens |
|---|---:|
| Minimum | 5 |
| Mean | 100.9776 |
| P50 | 90 |
| P75 | 119 |
| P90 | 219 |
| P95 | 249 |
| P99 | 286 |
| Maximum | 523 |

| Candidate | Covered | Coverage | Requires truncation | Truncation rate |
|---:|---:|---:|---:|---:|
| 64 | 9559 | 33.4383% | 19028 | 66.5617% |
| 128 | 22535 | 78.8295% | 6052 | 21.1705% |
| 256 | 27499 | 96.1941% | 1088 | 3.8059% |
| 512 | 28585 | 99.9930% | 2 | 0.0070% |

Selected `max_length = 256`.

Selected the smallest candidate among 128/256/512 that retains at least 95% of records. Observed P90=219, P95=249, P99=286; 96.1941% fit without truncation. This limits attention cost while remaining within DistilBERT's 512-token positional limit.

Persisted records remain variable length. Phase 2.8 should apply dynamic padding per batch.
