# Environment

The environment is one uv virtual environment with Python 3.12, the pins of `lab/requirements.txt`, and ALFWorld's data in `~/.cache/alfworld`. The Mac steps below describe what built `.venv`. The Colab L4 steps have never run on a VM, and CUDA has never run in this environment.

## Pins

`uv pip freeze --python .venv/bin/python` on 2026-11-04 gives these versions, with Python 3.12.13 and uv 0.11.7. The first five are what `uv pip tree --depth 0` lists as top level. The others are dependencies that `lab/src` imports or names by version.

| Package | Version |
| --- | --- |
| aiohttp | 3.14.5 |
| alfworld | 0.4.2 |
| fast-downward-textworld | 20.6.4 |
| kev | 0.1.0, git commit `2074cc44256200bb37540103fa4572901faa10f3` of `github.com/jaredpalmer/kev` |
| pytest | 9.2.0 |
| numpy | 2.5.4 |
| peft | 0.21.2 |
| pyyaml | 6.0.3 |
| textworld | 1.7.0 |
| torch | 2.8.0 |
| transformers | 5.22.1 |

Kev requires Python at least 3.12 and under 3.14, and torch at least 2.6 and under 2.9, which its `METADATA` file states.

Three checks were run on these pins on the Mac, and none of them downloaded a package.

- `uv pip install --dry-run --python .venv/bin/python -r lab/requirements.txt` printed "Would make no changes".
- `uv pip compile lab/requirements.txt --python-version 3.12` resolved 100 packages, and the list has no difference from `.venv`'s freeze.
- The same compile with `--python-platform x86_64-unknown-linux-gnu` resolved, with torch 2.8.0, the `nvidia-*-cu12` packages of CUDA 12.8, and triton 3.4.0. A resolve shows that the versions fit together and does not show that they run on a VM.

## This Mac

The machine is an Apple M4 with 10 cores and 24 GB of memory, on macOS 26.6.2. Torch reports no CUDA and MPS available.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r lab/requirements.txt
.venv/bin/alfworld-download
nice -n 10 .venv/bin/python -m pytest -q lab/src/tests
```

`.venv` takes 904 MB. `alfworld-download` puts 2.2 GB into `~/.cache/alfworld` (178 MB of it a detector file that nothing here uses, and the script has no flag to skip it). `ALFWORLD_DATA` moves that folder.

Halil's limits apply to every run here: the machine stays under 80% of memory, a process stays under 8 GB, at most 4 worker processes run, and anything over a minute runs under `nice -n 10`. The harness alone fits in them, with a largest worker of 379 MB in the 140-game expert run at 4 workers. Kev-0.8B is the tight part, and `lab/src/sys1/README.md` holds its measurements.

## Colab L4

These steps have not been run from this project. Colab's preinstalled Python and torch are unknown here, so the first command prints them and the steps build a separate environment that does not depend on them.

```bash
python -c "import sys, torch; print(sys.version, torch.__version__, torch.version.cuda)"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 /content/venv
uv pip install --python /content/venv/bin/python -r lab/requirements.txt
/content/venv/bin/alfworld-download
/content/venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
/content/venv/bin/python -m pytest -q lab/src/tests
```

Weights are fetched once, before the run, and the run then uses `HF_HUB_OFFLINE=1`. The job for Sys1 alone is the same on each machine, and only `--device` and `--workers` change.

```bash
/content/venv/bin/python lab/jobs/eval-sys1-alone.py --sys1 <checkpoint> --menu <menu file> --games <game list> --seed 0 --run <run> --device cuda --batch-size 8 --workers 4
```

`lab/jobs/arm-classification.py` installs less: a separate Python 3.12 environment under `/content/venv` with Kev, `torch==2.8.0`, `transformers==5.22.1`, `peft==0.21.2` and `numpy==2.5.4`, and no ALFWorld. Installing into Colab's own Python failed on 2026-11-04, because Kev's bound moved Colab's torch 2.11 to 2.8 and left torchvision, torchaudio and torchao behind.

## Fast kernels for Kev

Kev has fast CUDA kernels through `flash-linear-attention`, whose import name is `fla`. They are not in the pins, are not installed here, and have not been built or run in this project. Without them Kev's layers run through the reference PyTorch path, and Kev's runtime warns that this is much slower.

- `kev/fused_qwen35.py` requires exactly `flash-linear-attention==0.5.2` (`FLA_VERSION`), and raises an error at any other version.
- `fused_available()` is False without the package. `LoadOptions.fused=None` means off, so `KevSys1.load` runs the reference layers. `kev.serve` turns them on for CUDA, and `KEV_FUSED=0` declines them.
- `lib/execution-plan.md` rules out source builds of CUDA extensions on a VM and `torch.compile` on L4. Whether the kernels compile at first call on an L4 is Open.
- Resolving `flash-linear-attention==0.5.2` with the pins on Linux gives `fla-core` 0.5.2 and triton 3.4.0. Adding `triton>=3.7.1`, which `lab/jobs/arm-classification.py` installs as a separate step as Kev's `modal_app.py` does, has no solution in one resolve, because torch 2.8.0 pins triton to exactly 3.4.0 on Linux x86_64. The separate step therefore moves triton off torch's pin if Colab's torch has that pin, and no VM has run it.

## Game lists

A game list is a text file with one uid per line, or a JSON list, and `alf.env.load_game_list` reads both. `alf.env.list_games` gives a split's games in Bonn's order.

```bash
PYTHONPATH=lab/src .venv/bin/python -c "from alf.env import list_games; print('\n'.join(g.uid for g in list_games('valid_seen')))" > valid_seen.txt
```

`alf.env` refuses `valid_unseen`, the test games, unless a caller passes `allow_test_games=True`, and `lab/jobs/eval-sys1-alone.py` never passes it.
