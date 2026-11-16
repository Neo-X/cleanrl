| Environment | Checkpoint | Online agent (argmax Q) | Top-5% of D | Replay of best trajectory | DT (short context) | DT (20-step context) | CQL |
|---|---|---|---|---|---|---|---|
| MinAtar SpaceInvaders | 12.5M | 95 ± 43 | 151 ± 16 | 298 ± 43 | 54 ± 37 | 121 ± 25 | 113 ± 46 |
| MinAtar SpaceInvaders | 25M | 143 ± 13 | 150 ± 20 | 366 ± 51 | 90 ± 36 | 125 ± 20 | 106 ± 34 |
| MinAtar Breakout | 12.5M | 61 ± 17 | 42 ± 7 | 70 ± 11* | 44 ± 10 | 44 ± 8 | 30 ± 33 |
| MinAtar Breakout | 25M | 78 ± 17 | 48 ± 6 | 76 ± 12* | 47 ± 8 | 48 ± 6 | 39 ± 15 |
| Atari Asterix | 12.5M | 54 ± 12 | 115 ± 19 | 139 ± 26 | 44 ± 12 | 48 ± 17 | 21 ± 14 |
| Atari Asterix | 25M | 69 ± 12 | 198 ± 16 | 218 ± 5 | 43 ± 8 | 64 ± 13 | 23 ± 9 |

**Table R1.** Fixed-data interventions trained on the DQN replay buffer saved at 12.5M and 25M steps. Mean ± std over 5 collect seeds, one offline run per dataset. Each policy is evaluated with its argmax action in a deterministic environment (no sticky actions for Atari), with episodes capped at 1000 steps. *Replay of best trajectory* replays, open loop, the highest-return action sequence generated up to that checkpoint. \*Breakout shows the best episode in D instead, since these runs did not log the replay metric. Asterix returns are in clipped-reward units. All data was collected with ε = 0.1. In Breakout, one random action usually loses the ball, which is why the top 5% of D is below the greedy online agent.
