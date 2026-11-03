# Register card

The voice this repository writes in, and the rulings that settle a disputed
rewrite. `references/humanize.md` reads this file before every register pass,
`lint.py` reads its shapes and its overrides, and a proposal that breaks a line
here is rejected without being argued. One line per ruling, each with its date
and the person who made it.

## Target register

FILL. One paragraph, in the words a reviewer of this venue would use. Name the
sentence shape the venue expects, the verbs that belong to the field, and the
things this paper measures. The EuroSys form of this paragraph reads: balanced
sentences that carry their causality in compound clauses, mechanisms described
as mechanisms, design choices justified as trade-offs, quantitative findings
reported neutrally, and cryptographic terminology kept precise.

## The people

| Handle | Name | What they mark |
| --- | --- | --- |
| FILL | FILL | FILL, in their own words, from the comment record |

## Bans

One line per ban, with its date and whose it is. These override every rule in
`writing.md` and every suggestion an external model makes.

- FILL, YYYY-MM-DD, <handle>.

## Calibration rulings

The author's answers to the questions a register pass keeps asking. One line
each, with the date.

- A term this paper coins is defined with "We call", "which we call" or
  "called", varied. A term the field owns is "known as", or defined plainly.
  Never "we call" for a field term.
- A long sentence splits into two, never three.
- An announced list keeps its count and its ordinals.
- A pointer to another section varies its form and stays rare.
- FILL.

## Terms

| This paper coins | The field owns | Never |
| --- | --- | --- |
| FILL | FILL | FILL |

## lint shapes

One bullet per sentence shape a reader found and no rule catches. Each is a
Python regular expression and a short name, separated by two spaces. `lint.py`
compiles them and reports them under the name, so the next instance is
mechanical rather than a second reading.

- FILL  `\b(costs|needs|removes|fires|wakes)\s+(no|nothing)\b`  negation-predicate

## Overrides

A PI instruction that collides with a rule. The linter stops firing on it and
the register pass stops proposing against it. Quote the instruction.

| Rule | Instruction, quoted | Who, when |
| --- | --- | --- |
| FILL | FILL | FILL |

## External models

The two models that read the sentences, by the name they are invoked under.
Two of them, independently, because one model's reading agrees with another's
on about two fifths of the sentences.

- FILL
- FILL

## Thresholds

| Name | Value | Where it came from |
| --- | --- | --- |
| sentence stop, words | 30 | the author's standing instruction |
| telegraphic share ceiling, per cent | 7 | the EuroSys voice round, measured 11.5 before and 1.4 after |
| replacement growth ceiling, per cent | 20 | the PI's note that rephrasing gets too wordy |
