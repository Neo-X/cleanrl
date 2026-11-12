| Checkpoint | Online agent, argmax Q | Top-5% of D | DT | CQL |
|---|---|---|---|---|
| 12.5M | 389 ± 549 | 151 ± 16 | 133 ± 24 | 198 ± 172 |
| 25M | 452 ± 512 | 150 ± 20 | 131 ± 14 | 91 ± 25 |

Mean ± std over 5 collect seeds; each seed's value is the mean over 3 offline seeds. Returns are single deterministic evaluation episodes (argmax policy, deterministic env).
