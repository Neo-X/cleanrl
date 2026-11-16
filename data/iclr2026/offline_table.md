| Checkpoint | Online agent, argmax Q | Top-5% of D | Replay of best trajectory | DT | CQL |
|---|---|---|---|---|---|
| 12.5M | 95 ± 43 | 151 ± 16 | 298 ± 43 | 121 ± 25 | 113 ± 46 |
| 25M | 143 ± 13 | 150 ± 20 | 366 ± 51 | 125 ± 20 | 106 ± 34 |

Mean ± std over 5 collect seeds; one offline seed per collect seed. Returns are single deterministic evaluation episodes (argmax policy, deterministic env).
