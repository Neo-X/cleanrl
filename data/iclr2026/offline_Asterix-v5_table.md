| Checkpoint | Online agent, argmax Q | Top-5% of D | Replay of best trajectory | DT | CQL |
|---|---|---|---|---|---|
| 12.5M | 54 ± 12 | 115 ± 19 | 139 ± 26 | 48 ± 17 | 21 ± 14 |
| 25M | 69 ± 12 | 198 ± 16 | 218 ± 5 | 64 ± 13 | 23 ± 9 |

Mean ± std over 5 collect seeds; one offline seed per collect seed. Returns are single deterministic evaluation episodes (argmax policy, deterministic env).
