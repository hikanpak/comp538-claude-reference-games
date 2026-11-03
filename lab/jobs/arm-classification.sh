#!/usr/bin/env bash
# Mac side of arm-classification: pack the code, read MAX_STATE from the export's report, and run the job on an L4 through
# colab_job.sh. This starts a real Colab VM and spends compute units, charged to sys1.
#
#   lab/jobs/arm-classification.sh SEED [speedtest|full]
#
# Inputs, overridable by environment: TRAIN_JSONL and DEV_JSONL (from lab/src/train/export_kev.py, the report beside TRAIN_JSONL),
# RESULTS_DIR, PACK_DIR, TIMEOUT seconds, and EPOCHS LR BATCH ACCUM SPEED_ROWS PARTIAL_EVERY_MIN EVAL_BATCH INSTALL_FLA, passed through when set.
# Outputs: lab/results/arm-classification/seedN.json, and for speedtest .../speedtest/seedN.json, which sits one level down so that
# check_gates.py trained does not read a speed test as a trained run. Weights and partial weights go under .../weights/, which
# .gitignore skips.
set -euo pipefail

seed=${1:?usage: arm-classification.sh SEED [speedtest|full]}
mode=${2:-speedtest}
[[ $seed =~ ^[0-9]+$ ]] || { echo "seed must be a number, got '$seed'" >&2; exit 2; }
[[ $mode == speedtest || $mode == full ]] || { echo "mode must be speedtest or full, got '$mode'" >&2; exit 2; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train=${TRAIN_JSONL:-$root/lab/data/sys1-kev/train.jsonl.gz}
dev=${DEV_JSONL:-$root/lab/data/sys1-kev/dev_heldout_rooms.jsonl.gz}
report=${train%.gz}; report=${report%.jsonl}.report.json
for f in "$train" "$dev" "$report"; do [[ -f $f ]] || { echo "missing $f, run lab/src/train/export_kev.py first" >&2; exit 2; }; done

python=${PYTHON:-python3}
max_state=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["tokens"]["max_state_needed"])' "$report") \
  || { echo "$report has no tokens.max_state_needed, rerun the export with its token report" >&2; exit 2; }

commit=$(git -C "$root" rev-parse HEAD)
dirty=0
git -C "$root" diff --quiet HEAD -- lab/src/sys1 lab/src/train lab/jobs || dirty=1
[[ -z $(git -C "$root" ls-files --others --exclude-standard -- lab/src/sys1 lab/src/train lab/jobs) ]] || dirty=1
if [[ $dirty == 1 && $mode == full && ${ALLOW_DIRTY:-0} != 1 ]]; then
  echo "lab/src/sys1, lab/src/train or lab/jobs differ from $commit: commit first, because a result must name the code that made it (ALLOW_DIRTY=1 overrides)" >&2
  exit 2
fi

results=${RESULTS_DIR:-$root/lab/results/arm-classification}
stem=seed$seed
pack=${PACK_DIR:-$root/lab/data/sys1-kev}/code-$mode-$stem.tar.gz
mkdir -p "$(dirname "$pack")"
COPYFILE_DISABLE=1 tar -czf "$pack" --exclude '__pycache__' -C "$root/lab/src" sys1 train

args=(-c sys1 -g L4 -s "$root/lab/jobs/arm-classification.py"
      -u "$train:/content/train.jsonl.gz" -u "$dev:/content/dev.jsonl.gz" -u "$pack:/content/sys1_code.tar.gz"
      -e "SEED=$seed" -e "MODE=$mode" -e "MAX_STATE=$max_state" -e "GIT_COMMIT=$commit" -e "GIT_DIRTY=$dirty")
for key in EPOCHS LR BATCH ACCUM SPEED_ROWS PARTIAL_EVERY_MIN EVAL_BATCH INSTALL_FLA; do
  if [[ -n ${!key:-} ]]; then args+=(-e "$key=${!key}"); fi
done

if [[ $mode == speedtest ]]; then
  args+=(-n "arm-classification-$stem-speedtest" -l "$results/speedtest/logs/$stem.log" -t "${TIMEOUT:-2400}"
         -d "/content/out/$stem.json:$results/speedtest/$stem.json")
else
  args+=(-n "arm-classification-$stem" -l "$results/logs/$stem.log" -t "${TIMEOUT:-25200}"
         -d "/content/out/$stem.json:$results/$stem.json" -d "/content/out/$stem.tar:$results/weights/$stem.tar"
         -p "/content/out/$stem-partial.tar:$results/weights/$stem-partial.tar:300")
fi

exec "$root/lab/jobs/colab_job.sh" "${args[@]}"
