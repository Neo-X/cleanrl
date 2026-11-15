| Checkpoint | Online agent, argmax Q | Top-5% of D | DT | CQL |
|---|---|---|---|---|
| 12.5M | 227 ± 347 | 42 ± 7 | 44 ± 8 | 32 ± 32 |
| 25M | 558 ± 414 | 48 ± 6 | 48 ± 6 | 207 ± 365 |

Mean ± std over 5 collect seeds; one offline seed per collect seed. Returns are single deterministic evaluation episodes (argmax policy, deterministic env).
