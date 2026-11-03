# Ingest log

Anything that arrives in a format the repo does not use is dropped in `sink/`.
Nothing reads from it. `ingest.py` moves each file to its one target and adds a
row here. A file leaves the sink only when its target exists in the format.

| Date | Original | Target | Created |
| --- | --- | --- | --- |
