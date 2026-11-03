# Gates: Sys1 training set, Sys1 trained alone, then Gemma only if it passes

OWNS: lab/**, lib/**, CONTEXT.md, CLAUDE.md, GATES.md, plan/**

Scope: from Halil's instruction of 2026-11-03, a training set for Sys1, a Kev-0.8B trained for ALFWorld and measured against `lib/sys1-acceptance.md`, everything logged, and a report. Gemma runs only if Sys1 passes.

- [x] G1: project knowledge sits in the research layout with no placeholder
  CHECK: python3 -I lab/tools/check_gates.py knowledge
  EXPECT: knowledge recorded
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/a90/Documents/COURSES/FALL2026/Comp538/project; path=adefdc66accf/24 entries; EXPECT=matched; output-sha256=99517015e05396e114628976eeb1d07c6703b88c931e6d4053eaab5d1a5d9487; output-bytes=19

- [x] G2: every review finding on the execution path has a recorded decision
  CHECK: python3 -I lab/tools/check_gates.py critique
  EXPECT: critique recorded with decisions
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/a90/Documents/COURSES/FALL2026/Comp538/project; path=adefdc66accf/24 entries; EXPECT=matched; output-sha256=d0afd66c6295e95a30817d2ca02506b4fc6a6ec754a4bcb0e910303fa3928d65; output-bytes=33

- [ ] G3: the training set exists, its hashes hold, no test game and no room overlap
  CHECK: python3 -I lab/tools/check_gates.py dataset
  EXPECT: training set verified
  EVIDENCE: pending

- [ ] G4: the harness, adapter and dataset tests pass in the project environment
  CHECK: python3 -I lab/tools/check_gates.py tests
  EXPECT: code tests passed
  EVIDENCE: pending

- [ ] G5: the acceptance gates were committed before the first training result
  CHECK: python3 -I lab/tools/check_gates.py gates_first
  EXPECT: gates predate training
  EVIDENCE: pending

- [ ] G6: Sys1 has a trained classification run with a recorded seed
  CHECK: python3 -I lab/tools/check_gates.py trained
  EXPECT: sys1 trained
  EVIDENCE: pending

- [ ] G7: Sys1's gates are evaluated on 3 seeds, pass or fail
  CHECK: python3 -I lab/tools/check_gates.py verdict
  EXPECT: acceptance evaluated
  EVIDENCE: pending

- [x] G8: Colab spend stays within the caps, and no Gemma spend precedes a Sys1 pass
  CHECK: python3 -I lab/tools/check_gates.py ledger
  EXPECT: ledger within caps
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/a90/Documents/COURSES/FALL2026/Comp538/project; path=adefdc66accf/24 entries; EXPECT=matched; output-sha256=258c3b76dc3d51b23255da52646f8040ac1bb5305d36d8bfa45dcaa03dcad372; output-bytes=38

- [ ] G9: a report for Halil exists with the verdict, what was spent and what was not done
  CHECK: python3 -I lab/tools/check_gates.py report
  EXPECT: report written
  EVIDENCE: pending
