#!/usr/bin/env bash
# Mac side of arm-rl-model-in-control: pack the code and run the job on an L4 through colab_job.sh. This starts a real Colab VM and spends compute units, charged to sys1.
# No agent has run it, and no VM has run the job's install steps.
#
#   lab/jobs/arm-rl-model-in-control.sh SEED [speedtest|full]
#
# Inputs, overridable by environment: TRAIN_JSONL DEV_JSONL MENU TRAIN_GAMES DEV_GAMES, INIT_TAR (the classification arm's checkpoint tar), RESULTS_DIR, PACK_DIR, TIMEOUT seconds.
# STEP_BUDGET is required in full mode, from a speed test. Every setting in `pass` below goes through when set. REWARD picks R3 or R4, R3 by default.
# Resume: when <results>/weights/<stem>-state.tar exists it is uploaded and the job continues from it. RESUME=0 refuses to start over it, and FRESH=1 ignores it.
# Outputs: <results>/<stem>.json, and for a speed test <results>/speedtest/<stem>.json, which sits one level down so nothing reads it as a trained run. <results>/curves/<stem>-curve.json. Weights under
# <results>/weights/, which .gitignore skips. The final checkpoint is a pull and not a download, because a run stopped at MAX_SECONDS has none yet and colab_job.sh would call that a failure, so read `completed` in the result.
set -euo pipefail

topic=arm-rl-model-in-control
seed=${1:?usage: $topic.sh SEED [speedtest|full]}
mode=${2:-speedtest}
[[ $seed =~ ^[0-9]+$ ]] || { echo "seed must be a number, got '$seed'" >&2; exit 2; }
[[ $mode == speedtest || $mode == full ]] || { echo "mode must be speedtest or full, got '$mode'" >&2; exit 2; }
if [[ $mode == full && -z ${STEP_BUDGET:-} ]]; then echo "full mode needs STEP_BUDGET, from the speed test" >&2; exit 2; fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
reward=${REWARD:-R3}
[[ $reward == R3 || $reward == R4 ]] || { echo "REWARD must be R3 or R4, got '$reward'" >&2; exit 2; }
stem=seed$seed-$reward
pass=(REWARD FALLBACK STEP_BUDGET CLIP LR LR_WARMUP LR_TOTAL_UPDATES TRANSITIONS_PER_ITER MINIBATCH PPO_EPOCHS GAMMA LAM ACTOR_COEF VF_COEF ENTROPY_COEF VALUE_LR VALUE_DETACH KL KL_COEF KL_ROWS KL_POOL_ROWS KL_GUARD
      K TRACE_WINDOW MAX_STEPS GAMES_PER_COLLECT BATCH_GAMES NUM_WORKERS MICRO_BATCH EVAL_BATCH DEV_EVERY_STEPS DEV_GAMES_LIMIT CHECKPOINT_EVERY_MIN MAX_SECONDS INSTALL_FLA)
train=${TRAIN_JSONL:-$root/lab/data/sys1-kev/train.jsonl.gz}
dev=${DEV_JSONL:-$root/lab/data/sys1-kev/dev_heldout_rooms.jsonl.gz}
menu=${MENU:-$root/lab/results/menu-statistics/menu.json}
train_games=${TRAIN_GAMES:-$root/lab/results/sys1-training-set/train.txt}
dev_games=${DEV_GAMES:-$root/lab/results/sys1-training-set/dev_heldout_rooms.txt}
init=${INIT_TAR:-$root/lab/results/arm-classification/weights/seed$seed.tar}
for f in "$train" "$dev" "$menu" "$train_games" "$dev_games" "$root/lab/src/menu.py"; do [[ -f $f ]] || { echo "missing $f" >&2; exit 2; }; done

results=${RESULTS_DIR:-$root/lab/results/$topic}
state=$results/weights/$stem-state.tar
if [[ -f $state && ${FRESH:-0} != 1 && ${RESUME:-1} == 0 ]]; then echo "$state exists: delete it, or use FRESH=1 to start over" >&2; exit 2; fi
if [[ ! -f $state || ${FRESH:-0} == 1 ]]; then
  [[ -f $init ]] || { echo "missing $init: RL starts from the classification arm's weights" >&2; exit 2; }
fi

commit=$(git -C "$root" rev-parse HEAD)
dirty=0
git -C "$root" diff --quiet HEAD -- lab/src lab/jobs || dirty=1
[[ -z $(git -C "$root" ls-files --others --exclude-standard -- lab/src lab/jobs) ]] || dirty=1
if [[ $dirty == 1 && $mode == full && ${ALLOW_DIRTY:-0} != 1 ]]; then
  echo "lab/src or lab/jobs differ from $commit: commit first, because a result must name the code that made it (ALLOW_DIRTY=1 overrides)" >&2
  exit 2
fi

pack=${PACK_DIR:-$root/lab/data/sys1-kev}/code-$topic-$mode-$stem.tar.gz
mkdir -p "$(dirname "$pack")"
COPYFILE_DISABLE=1 tar -czf "$pack" --exclude '__pycache__' -C "$root/lab/src" alf harness sys1 train menu.py -C "$root/lab" requirements.txt

args=(-c sys1 -g L4 -s "$root/lab/jobs/$topic.py"
      -u "$train:/content/train.jsonl.gz" -u "$dev:/content/dev.jsonl.gz" -u "$menu:/content/menu.json" -u "$train_games:/content/train_games.txt" -u "$dev_games:/content/dev_games.txt" -u "$pack:/content/sys1_code.tar.gz"
      -e "SEED=$seed" -e "MODE=$mode" -e "STEM=$stem" -e "GIT_COMMIT=$commit" -e "GIT_DIRTY=$dirty")
if [[ -f $state && ${FRESH:-0} != 1 ]]; then args+=(-u "$state:/content/resume.tar"); else args+=(-u "$init:/content/init.tar"); fi
for key in "${pass[@]}"; do
  if [[ -n ${!key:-} ]]; then args+=(-e "$key=${!key}"); fi
done

if [[ $mode == speedtest ]]; then
  args+=(-n "$topic-$stem-speedtest" -l "$results/speedtest/logs/$stem.log" -t "${TIMEOUT:-3600}" -d "/content/out/$stem.json:$results/speedtest/$stem.json")
else
  # 28500 s, under colab_job.sh's limit of 28800. The job stops training at MAX_SECONDS, 27000 s, and then finishes its last dev point, refit and checkpoint.
  args+=(-n "$topic-$stem" -l "$results/logs/$stem.log" -t "${TIMEOUT:-28500}" -d "/content/out/$stem.json:$results/$stem.json"
         -p "/content/out/$stem-state.tar:$state:300" -p "/content/out/$stem-curve.json:$results/curves/$stem-curve.json:300" -p "/content/out/$stem.tar:$results/weights/$stem.tar:300")
fi

exec "${COLAB_JOB:-$root/lab/jobs/colab_job.sh}" "${args[@]}"
