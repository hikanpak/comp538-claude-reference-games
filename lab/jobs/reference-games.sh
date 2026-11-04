#!/usr/bin/env bash
# Setup for reference-games: the Python environment, then the games from this repository's data branch.
#
#   lab/jobs/reference-games.sh dev|test
#
# `dev` fetches the 308 dev games and `test` the 134 test games, so a session holds only the games it plays. It prints SETUP_OK at the end.
# The planner package has no Linux wheel, so it compiles here and needs a C++ compiler and cmake.
set -euo pipefail
which=${1:?usage: reference-games.sh dev|test}
[[ $which == dev || $which == test ]] || { echo "the argument is dev or test, got '$which'" >&2; exit 2; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
data=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
python=$root/.reference-games-venv/bin/python

command -v uv >/dev/null || python3 -m pip install --quiet --user --break-system-packages uv
uv=$(command -v uv || echo "$HOME/.local/bin/uv")
[[ -x $python ]] || "$uv" venv --python 3.12 "$root/.reference-games-venv"
"$uv" pip install --quiet --python "$python" alfworld==0.4.2 textworld==1.7.0 fast-downward-textworld==20.6.4 numpy==2.5.4 pyyaml==6.0.3 aiohttp==3.14.5

mkdir -p "$data"
git -C "$root" fetch --quiet --depth 1 origin "data-$which-game-files"
git -C "$root" show "FETCH_HEAD:$which-game-files.tar.gz" | tar -xz -C "$data"
echo SETUP_OK
