| Checkpoint | Online agent, argmax Q | Top-5% of D | Best episode in D | DT | CQL |
|---|---|---|---|---|---|
| 12.5M | 61 ± 17 | 42 ± 7 | 70 ± 11 | 44 ± 8 | 30 ± 33 |
| 25M | 78 ± 17 | 48 ± 6 | 76 ± 12 | 48 ± 6 | 39 ± 15 |

Mean ± std over 5 collect seeds; one offline seed per collect seed. Returns are single deterministic evaluation episodes (argmax policy, deterministic env).
