"""The training set for Sys1 on ALFWorld. Run in this order, each step reading what the one before wrote:

    python -m dataset.splits      # the games of each split, through AlfredTWEnv's own filter, and the held-out rooms
    python -m dataset.generate    # plays the rollouts into the pool, resumable
    python -m dataset.menu_stats  # fits the menus on the pool's train games and measures them on the dev splits
    python -m dataset.build       # the core set, the dev splits and the manifest
    python lab/src/train/export_kev.py ...   # Kev's JSONL, see `lab/results/sys1-training-set/CARD.md`
    python -m dataset.verify_batch           # one real batch through kev.train's loader and the model
    python -m dataset.audit       # the audit file

`lab/results/sys1-training-set/CARD.md` says what each family is and what the set cannot teach.
"""
