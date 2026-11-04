# Menu statistics

Starting k is 20. This is Claude's choice from the numbers below and it departs from the start of 10 that Halil named, so it is his to overrule. The numbers in this section are rule `form` on `dev_heldout_rooms`, the split the gates read, unless a sentence says otherwise, and the tables below hold them.

At k = 10 the expert's command is on the menu at 51.7% of steps. Sys1 would have to fall back at about every second step, and two fallbacks in a row hand control to the LLM. The commands that finish a task rank low, because each form of them, with its object class and place class, is rarer than most `go to` forms. At k = 10, `clean`, `cool` and `heat` are on the menu at 0.0% of their steps, `take` at 15.9% and `move` at 18.6%. Only 13.1% of the expert's games keep every command on the menu, and no clean, cool or heat game does.

At k = 20 the expert's command is on the menu at 83.7% of steps, so the implied fallback rate is 16.3%, and 51.6% of the expert's games keep every command on the menu. Fallback stays a minority label with real examples, 17.8% of `core`. Coverage stops growing at this point. It is 83.8% at k = 25 and 85.5% at k = 30, and the next rise comes at k = 35, where a menu averages 27.1 commands against 28.4 admissible, which is no longer a short list. With all admissible commands nothing is off the menu, so fallback has no positive label and the gate on falling back when it should cannot be measured. k = 5 holds 26.3%. On `dev` the same rule holds 53.0% at k = 10 and 82.4% at k = 20. The weaker rule, ranking by command kind alone, holds 25.6% at k = 10 and 51.0% at k = 20 on `dev_heldout_rooms`.

The cost of 20 over 10 is a longer choice for Sys1, a mean of 18.4 commands against 10, so every row has more options and more tokens. `speedtest-200.jsonl.gz` has rows of this size, so the job's speed test measures that cost.

k = 20 does not repair the low ranks. At k = 20, `cool` is on the menu at 0.0% of its steps and `heat` at 19.5%, and no cool or heat game keeps every command on the menu. All held-out cool and heat games are in room 7. The 51.6% sits below the 54.5 of the gate on solving games, which masks fallback, so Sys1 can only pick from the menu there. A cool game cannot be won without `cool`, so the 51.6% counts only the expert's path, and Sys1 could win another way only where the command it needs comes onto the menu at another state (Heuristic).

Whether to change the menu rule, k or that gate is Halil's decision. None of them needs new rollouts: `dataset.menu_stats`, `dataset.build` and the export rerun from the pool.

Empirical. Every number comes from `stats.json` beside this file, rollouts of seed 0, expert under `tracking=always`, git commit `57a8ea2e53fe537e36e74f1346b21fdfe2060030` (dirty: True). A row counts when its rollout was won and its label is usable. Coverage is the share of steps with the expert's command on the menu, and the implied fallback rate is one minus coverage. `form` is the plan's rule and `kind` the weaker one.

## dev: 140 games, 15383 rows

Admissible commands per step: mean 29.9, maximum 71.

| Rule | k | Coverage % | Implied fallback % | `go to` share of menu slots % | Expert games with every label on the menu % |
| --- | --- | --- | --- | --- | --- |
| form | 5 | 26.2 | 73.8 | 91.3 | 0.0 |
| form | 10 | 53.0 | 47.0 | 88.3 | 7.8 |
| form | 20 | 82.4 | 17.6 | 86.4 | 43.8 |
| form | all | 100.0 | 0.0 | 82.6 | 100.0 |
| kind | 5 | 8.2 | 91.8 | 100.0 | 0.0 |
| kind | 10 | 19.9 | 80.1 | 98.6 | 3.0 |
| kind | 20 | 51.0 | 49.0 | 91.2 | 39.8 |
| kind | all | 100.0 | 0.0 | 82.6 | 100.0 |

Coverage % by family, rule `form`:

| Family | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| expert-path | 28.6 | 57.9 | 85.4 | 100.0 |
| recovery | 22.7 | 47.4 | 78.7 | 100.0 |
| loop-break | 29.5 | 55.2 | 84.9 | 100.0 |

Coverage % by family, rule `kind`:

| Family | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| expert-path | 7.0 | 18.3 | 48.4 | 100.0 |
| recovery | 8.6 | 20.1 | 51.2 | 100.0 |
| loop-break | 10.6 | 23.6 | 57.2 | 100.0 |

Coverage % by task type, rule `form`:

| Task type | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| look_at_obj_in_light | 34.7 | 61.9 | 91.5 | 100.0 |
| pick_and_place_simple | 26.1 | 59.1 | 87.4 | 100.0 |
| pick_clean_then_place_in_recep | 23.9 | 53.6 | 84.1 | 100.0 |
| pick_cool_then_place_in_recep | 21.2 | 41.3 | 70.4 | 100.0 |
| pick_heat_then_place_in_recep | 20.1 | 44.6 | 73.1 | 100.0 |
| pick_two_obj_and_place | 35.4 | 59.2 | 89.4 | 100.0 |

Coverage % by the kind of the expert's command, rule `form`, with how many rows have that kind:

| Command kind | Rows | k=5 | k=10 | k=20 |
| --- | --- | --- | --- | --- |
| go to | 7786 | 42.6 | 69.0 | 94.3 |
| open | 2282 | 13.2 | 55.5 | 90.6 |
| close | 1519 | 17.0 | 52.9 | 86.4 |
| move | 1378 | 0.0 | 10.9 | 47.7 |
| take | 1287 | 0.0 | 9.0 | 49.8 |
| look | 400 | 34.0 | 84.0 | 97.0 |
| clean | 214 | 0.0 | 13.1 | 38.3 |
| cool | 196 | 0.0 | 0.0 | 5.1 |
| use | 139 | 14.4 | 60.4 | 95.7 |
| heat | 126 | 0.0 | 0.0 | 21.4 |
| examine | 56 | 0.0 | 0.0 | 42.9 |

Rule `form` along a finer range of k. Coverage counts all families, the expert columns count the expert's own won games, and a game counts as whole when every usable label of it is on the menu at its state. Those games are the ones a Sys1 restricted to the menu could finish by following the expert:

| k | Mean menu length | Coverage % | Expert-path coverage % | Whole expert games % | Whole look_at_obj_in_light % | Whole pick_and_place_simple % | Whole pick_clean_then_place_in_recep % | Whole pick_cool_then_place_in_recep % | Whole pick_heat_then_place_in_recep % | Whole pick_two_obj_and_place % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 5.0 | 26.2 | 28.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | 8.0 | 45.3 | 50.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 10 | 10.0 | 53.0 | 57.9 | 7.8 | 30.8 | 8.6 | 5.1 | 0.0 | 0.0 | 9.7 |
| 12 | 12.0 | 60.2 | 64.3 | 16.8 | 38.5 | 20.0 | 20.5 | 0.0 | 0.0 | 24.2 |
| 15 | 14.8 | 71.9 | 76.2 | 28.5 | 53.8 | 45.7 | 21.8 | 0.0 | 0.0 | 45.2 |
| 18 | 17.4 | 79.4 | 82.9 | 38.0 | 61.5 | 65.7 | 30.8 | 0.0 | 0.0 | 56.5 |
| 20 | 18.9 | 82.4 | 85.4 | 43.8 | 76.9 | 71.4 | 30.8 | 0.0 | 0.0 | 74.2 |
| 22 | 20.3 | 85.3 | 87.8 | 50.7 | 92.3 | 74.3 | 30.8 | 12.9 | 13.0 | 80.6 |
| 25 | 22.2 | 88.4 | 90.6 | 54.5 | 92.3 | 80.0 | 38.5 | 12.9 | 13.0 | 85.5 |
| 30 | 24.8 | 91.4 | 93.0 | 63.5 | 100.0 | 82.9 | 53.8 | 17.1 | 45.7 | 85.5 |
| 35 | 26.8 | 94.2 | 95.2 | 74.2 | 100.0 | 91.4 | 57.7 | 47.1 | 60.9 | 90.3 |
| 40 | 28.4 | 97.1 | 97.7 | 84.0 | 100.0 | 91.4 | 61.5 | 72.9 | 87.0 | 100.0 |
| all | 29.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

Share of random or repeated commands inserted into won rollouts that were on the menu, rule `form`, so a live Sys1 could have picked them %:

| Family | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| recovery | 20.6 | 44.4 | 76.4 | 100.0 |
| loop-break | 27.2 | 49.8 | 72.7 | 100.0 |

Share of admissible commands by kind %: go to 82.5, examine 4.7, inventory 3.3, look 3.3, take 3.2, open 1.2, move 0.8, close 0.5, cool 0.1, clean 0.1, heat 0.1, slice 0.1, use 0.1.

## dev_heldout_rooms: 168 games, 21387 rows

Admissible commands per step: mean 28.2, maximum 55.

| Rule | k | Coverage % | Implied fallback % | `go to` share of menu slots % | Expert games with every label on the menu % |
| --- | --- | --- | --- | --- | --- |
| form | 5 | 26.3 | 73.7 | 91.1 | 0.0 |
| form | 10 | 51.7 | 48.3 | 84.6 | 13.1 |
| form | 20 | 83.7 | 16.3 | 84.2 | 51.6 |
| form | all | 100.0 | 0.0 | 81.9 | 100.0 |
| kind | 5 | 11.1 | 88.9 | 100.0 | 0.0 |
| kind | 10 | 25.6 | 74.4 | 94.7 | 17.9 |
| kind | 20 | 51.0 | 49.0 | 88.9 | 51.6 |
| kind | all | 100.0 | 0.0 | 81.9 | 100.0 |

Coverage % by family, rule `form`:

| Family | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| expert-path | 28.4 | 55.1 | 85.3 | 100.0 |
| recovery | 22.2 | 46.5 | 81.2 | 100.0 |
| loop-break | 32.0 | 57.2 | 86.3 | 100.0 |

Coverage % by family, rule `kind`:

| Family | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| expert-path | 8.8 | 23.3 | 47.4 | 100.0 |
| recovery | 12.1 | 25.4 | 52.4 | 100.0 |
| loop-break | 14.7 | 32.7 | 57.8 | 100.0 |

Coverage % by task type, rule `form`:

| Task type | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| look_at_obj_in_light | 70.0 | 95.4 | 100.0 | 100.0 |
| pick_and_place_simple | 29.0 | 65.8 | 91.8 | 100.0 |
| pick_clean_then_place_in_recep | 29.1 | 61.0 | 89.7 | 100.0 |
| pick_cool_then_place_in_recep | 18.1 | 33.3 | 72.9 | 100.0 |
| pick_heat_then_place_in_recep | 18.4 | 37.5 | 76.0 | 100.0 |
| pick_two_obj_and_place | 27.3 | 59.9 | 90.5 | 100.0 |

Coverage % by the kind of the expert's command, rule `form`, with how many rows have that kind:

| Command kind | Rows | k=5 | k=10 | k=20 |
| --- | --- | --- | --- | --- |
| go to | 11702 | 40.2 | 67.3 | 91.3 |
| open | 2913 | 12.4 | 43.9 | 92.9 |
| close | 2035 | 12.9 | 33.4 | 86.3 |
| move | 1754 | 0.0 | 18.6 | 57.2 |
| take | 1648 | 0.0 | 15.9 | 59.5 |
| look | 498 | 29.5 | 100.0 | 100.0 |
| cool | 305 | 0.0 | 0.0 | 0.0 |
| heat | 210 | 0.0 | 0.0 | 19.5 |
| use | 144 | 100.0 | 100.0 | 100.0 |
| clean | 140 | 0.0 | 0.0 | 45.0 |
| examine | 38 | 0.0 | 5.3 | 73.7 |

Rule `form` along a finer range of k. Coverage counts all families, the expert columns count the expert's own won games, and a game counts as whole when every usable label of it is on the menu at its state. Those games are the ones a Sys1 restricted to the menu could finish by following the expert:

| k | Mean menu length | Coverage % | Expert-path coverage % | Whole expert games % | Whole look_at_obj_in_light % | Whole pick_and_place_simple % | Whole pick_clean_then_place_in_recep % | Whole pick_cool_then_place_in_recep % | Whole pick_heat_then_place_in_recep % | Whole pick_two_obj_and_place % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 5.0 | 26.3 | 28.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | 8.0 | 45.7 | 49.2 | 4.8 | 35.6 | 5.7 | 0.0 | 0.0 | 0.0 | 1.9 |
| 10 | 10.0 | 51.7 | 55.1 | 13.1 | 73.3 | 17.1 | 0.0 | 0.0 | 0.0 | 13.0 |
| 12 | 11.8 | 57.9 | 60.4 | 21.3 | 82.2 | 19.0 | 41.2 | 0.0 | 0.0 | 25.9 |
| 15 | 14.5 | 70.4 | 72.5 | 35.3 | 100.0 | 42.9 | 41.2 | 0.0 | 0.0 | 60.2 |
| 18 | 17.0 | 82.9 | 84.8 | 50.4 | 100.0 | 84.8 | 41.2 | 0.0 | 0.0 | 88.9 |
| 20 | 18.4 | 83.7 | 85.3 | 51.6 | 100.0 | 87.6 | 41.2 | 0.0 | 0.0 | 91.7 |
| 22 | 19.6 | 83.7 | 85.3 | 51.8 | 100.0 | 88.6 | 41.2 | 0.0 | 0.0 | 91.7 |
| 25 | 21.3 | 83.8 | 85.3 | 51.8 | 100.0 | 88.6 | 41.2 | 0.0 | 0.0 | 91.7 |
| 30 | 24.2 | 85.5 | 87.1 | 51.8 | 100.0 | 88.6 | 41.2 | 0.0 | 0.0 | 91.7 |
| 35 | 27.1 | 98.6 | 98.9 | 80.7 | 100.0 | 91.4 | 70.6 | 73.0 | 57.7 | 91.7 |
| 40 | 28.1 | 99.7 | 99.7 | 94.6 | 100.0 | 97.1 | 70.6 | 99.1 | 92.3 | 98.1 |
| all | 28.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

Share of random or repeated commands inserted into won rollouts that were on the menu, rule `form`, so a live Sys1 could have picked them %:

| Family | k=5 | k=10 | k=20 | k=all |
| --- | --- | --- | --- | --- |
| recovery | 25.1 | 49.2 | 78.9 | 100.0 |
| loop-break | 30.4 | 59.5 | 80.2 | 100.0 |

Share of admissible commands by kind %: go to 81.8, examine 4.9, take 3.6, inventory 3.5, look 3.5, open 1.1, move 0.7, close 0.4, cool 0.1, use 0.1, heat 0.1, clean 0.1, slice 0.0.

