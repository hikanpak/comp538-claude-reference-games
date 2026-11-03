# The installed handcoded expert has tracking faults, and the training set uses a repaired one

Opened 2026-11-04 by the local harness agent, checked at runtime on ALFWorld 0.4.2.

`update_state_tracking` looks for `put`, but ALFWorld 0.4.2's command is `move X to Y`, so `inventory` keeps
moved objects, which breaks the two-object task on the expert's own path. `go to` moves `curr_recep` even on
"Nothing happens". `action_backlog` keeps a stale `close X`.

Empirical, 140 `valid_seen` games, seed 0, in `lab/results/eval-sys1-alone/`:

| Measure | Shipped expert (`tracking=foreign`, teacher-only) | Repaired tracking (`tracking=always`) |
| --- | --- | --- |
| Success | 83.57 | 95.71 |
| Mean steps | 21.84 | 17.91 |
| Repeat rate | 0.0298 | 0.0056 |
| `label_random` share | 0.1073 | 0.0207 |
| Two-object task, 24 games | 0.25 | 0.917 |

Bonn's published expert is 90.3 on test with the shipped expert, and a teacher-only `foreign` game equalled
Bonn's stack command by command on 30 of 30 training games in a probe (no result file).

Decision, Claude's and not Halil's, for him to rule on: labels in the training set come from `tracking=always`,
because it is the better teacher and it is what a nice training set needs. The published-number gate on the
expert uses the shipped behaviour, so it stays comparable with Bonn.
