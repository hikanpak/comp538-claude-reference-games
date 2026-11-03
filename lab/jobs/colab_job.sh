#!/usr/bin/env bash
# Run one script on a Colab VM: allocate, check the GPU, upload inputs, execute, download outputs, release, write the ledger.
# The VM is released on every exit path, including failure, Ctrl-C, SIGTERM and the wrong GPU.
#
#   lab/jobs/colab_job.sh -c sys1|gemma -n NAME -s SCRIPT -l LOG [-g L4] [-u LOCAL:REMOTE]... [-d REMOTE:LOCAL]...
#                         [-e KEY=VALUE]... [-t SECONDS] [-p REMOTE:LOCAL:SECONDS]...
#
# -g takes a preference list, L4 by default. Each GPU type gets three allocation attempts 20 s apart, because Colab refuses
#    allocations under short capacity limits. After allocation nvidia-smi must name the allocated type, so the default
#    run holds only on an L4, and anything else exits 5. Types are limited to those whose CU rate this account has measured.
# -c names who pays, one of sys1 or gemma. It is required, and lab/tools/check_gates.py sums the ledger by it.
# -p pulls REMOTE to LOCAL every SECONDS while the script runs, through a temporary name, so a failed pull keeps the last good copy.
#    The colab CLI transfers single files only, so REMOTE is a file, for example a tar the script replaces atomically.
# -t bounds the script's wall time, 8 hours at most. colab exec's --timeout does not bound wall time (a compile run once hung
#    72 min past it), so perl's alarm kills the client at the limit. Upload, download, allocation and release have alarms too.
#
# Remote paths are absolute under /content, and an upload's parent directory must already exist there. The script runs in the
# Colab kernel, so it takes arguments from -e, not sys.argv. colab exec exits 0 even when the script raises (the CLI does
# not check the cell's status, colab_cli/commands/execution.py), so this wrapper appends one print with a fresh token to a copy
# of the script and treats a missing token as failure. A script must therefore end by falling off its last line, not by sys.exit.
#
# Exit codes: 0 done, 2 usage, 3 no allocation, 4 upload failed, 5 wrong GPU, 6 script did not reach its end, 7 a download failed,
# 8 the release failed and a VM may still be billing, 129 SIGHUP, 130 SIGINT, 142 wall-clock limit, 143 SIGTERM, others from colab exec.
# Ledger: one line per VM that may have existed, lab/results/colab-ledger.csv, columns when,job,gpu,wall_s,cu,charged_to.
set -uo pipefail

gpu=L4; name=""; script=""; log=""; timeout=7200; charged=""
ups=(); downs=(); envs=(); pulls=()
while getopts "g:n:s:l:u:d:e:t:c:p:" o; do
  case $o in
    g) gpu=$OPTARG ;; n) name=$OPTARG ;; s) script=$OPTARG ;; l) log=$OPTARG ;; t) timeout=$OPTARG ;; c) charged=$OPTARG ;;
    u) ups+=("$OPTARG") ;; d) downs+=("$OPTARG") ;; e) envs+=(--env "$OPTARG") ;; p) pulls+=("$OPTARG") ;;
    *) echo "bad option" >&2; exit 2 ;;
  esac
done

usage_error() { echo "colab_job: $*" >&2; exit 2; }
[[ -n $name && -n $script && -n $log ]] || usage_error "need -n NAME -s SCRIPT -l LOG"
[[ $name != *,* ]] || usage_error "a comma in -n would break the ledger: $name"
[[ -f $script ]] || usage_error "no such script: $script"
[[ $script == *.py ]] || usage_error "-s takes a .py file, the end-of-script check appends Python: $script"
[[ $charged == sys1 || $charged == gemma ]] || usage_error "-c must be sys1 or gemma, got '$charged'"
[[ $timeout =~ ^[0-9]+$ ]] && (( timeout >= 1 && timeout <= 28800 )) || usage_error "-t must be 1 to 28800 seconds, got '$timeout'"

# CU per hour, measured on Halil's account on 2026-10-28 (CONTEXT.md for L4; yargitay-gemma colab-gpu skill for G4 and A100).
rate_of() { case $1 in L4) echo 1.54 ;; A100) echo 5.30 ;; G4) echo 8.90 ;; *) return 1 ;; esac; }
# What nvidia-smi says for each type. A bare substring would let an L40S pass as an L4.
gpu_pattern() { case $1 in L4) echo '(^|[^[:alnum:]])L4([^[:alnum:]]|$)' ;; A100) echo '(^|[^[:alnum:]])A100([^[:alnum:]]|$)' ;; G4) echo 'RTX PRO 6000' ;; esac; }
for g in ${gpu//,/ }; do rate_of "$g" >/dev/null || usage_error "no CU rate known for GPU type '$g'"; done
for p in ${ups[@]+"${ups[@]}"}; do [[ -f ${p%%:*} ]] || usage_error "no such upload: ${p%%:*}"; done
for p in ${pulls[@]+"${pulls[@]}"}; do
  [[ ${p##*:} =~ ^[1-9][0-9]*$ && $p == *:*:* ]] || usage_error "-p needs REMOTE:LOCAL:SECONDS, got '$p'"
done

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ledger=${COLAB_LEDGER:-$root/lab/results/colab-ledger.csv}
retry_sleep=${COLAB_JOB_RETRY_SLEEP:-20}
upload_sleep=${COLAB_JOB_UPLOAD_SLEEP:-5}
cmd_limit=${COLAB_JOB_CMD_LIMIT:-1800}
check_limit=${COLAB_JOB_CHECK_LIMIT:-180}
stop_limit=${COLAB_JOB_STOP_LIMIT:-120}

mkdir -p "$(dirname "$log")"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/colab_job.XXXXXX") || usage_error "cannot make a temporary directory"
C=(colab --auth=adc)

stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$log"; }

child=""
pull_pids=()
# A remote command with a wall-clock alarm. It runs in the background so a signal handler runs at once, not after the child ends.
bounded() {
  local limit=$1; shift
  perl -e 'alarm shift; exec @ARGV' "$limit" "$@" >>"$log" 2>&1 &
  child=$!
  wait "$child"
  local rc=$?
  child=""
  return $rc
}

pull_once() {
  local remote=$1 dest=$2 part="$2.part"
  mkdir -p "$(dirname "$dest")"
  if bounded "$cmd_limit" "${C[@]}" download -s "$name" "$remote" "$part"; then
    mv -f "$part" "$dest"
    return 0
  fi
  rm -f "$part"
  return 1
}

pull_loop() {
  local remote=$1 dest=$2 every=$3 waiter=""
  trap '[[ -n $waiter ]] && kill "$waiter" 2>/dev/null; [[ -n $child ]] && kill "$child" 2>/dev/null; rm -f "$dest.part"; exit 0' TERM
  trap '' INT HUP
  while :; do
    sleep "$every" &
    waiter=$!
    wait "$waiter"
    waiter=""
    if pull_once "$remote" "$dest"; then
      echo "[$(date '+%F %T')] pulled $remote -> $dest" >>"$log"
    else
      echo "[$(date '+%F %T')] pull of $remote failed, keeping the last copy" >>"$log"
    fi
  done
}

stop_pulls() {
  local pid
  for pid in ${pull_pids[@]+"${pull_pids[@]}"}; do
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
  pull_pids=()
}

# Set before each allocation attempt and cleared after a refused one, so an interrupt during `colab new` still releases and bills.
maybe_vm=0; attempted=0; t0=""; when=""; rate=""; ledger_gpu=""

release() {
  local try
  for try in 1 2 3; do
    stamp "release $name try=$try"
    if bounded "$stop_limit" "${C[@]}" stop -s "$name"; then return 0; fi
    sleep "$upload_sleep"
  done
  return 1
}

write_ledger() {
  local wall=$(( $(date +%s) - t0 ))
  local cu
  cu=$(awk -v w="$wall" -v r="$rate" 'BEGIN { printf "%.4f", w / 3600 * r }')
  mkdir -p "$(dirname "$ledger")"
  [[ -s $ledger ]] || echo "when,job,gpu,wall_s,cu,charged_to" >>"$ledger"
  printf '%s,%s,%s,%s,%s,%s\n' "$when" "$name" "${ledger_gpu//,/;}" "$wall" "$cu" "$charged" >>"$ledger"
  stamp "ledger $name gpu=$ledger_gpu wall_s=$wall cu=$cu charged_to=$charged"
}

on_exit() {
  local rc=$?
  trap '' INT TERM HUP   # a second Ctrl-C must not cut the release short
  stop_pulls
  if [[ -n $child ]]; then kill "$child" 2>/dev/null; wait "$child" 2>/dev/null; child=""; fi
  if (( attempted )); then
    if ! release; then
      stamp "RELEASE FAILED for $name, A VM MAY STILL BE BILLING: run colab --auth=adc sessions and stop it"
      (( rc == 0 )) && rc=8
    fi
    (( maybe_vm )) && write_ledger
  fi
  rm -rf "$tmp"
  exit "$rc"
}

on_signal() {
  stamp "signal, exiting with $1"
  if [[ -n $child ]]; then kill "$child" 2>/dev/null; fi
  exit "$1"
}

trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

when=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
got=""
for g in ${gpu//,/ }; do
  for try in 1 2 3; do
    stamp "new $name gpu=$g try=$try"
    attempted=1; maybe_vm=1; t0=$(date +%s); ledger_gpu=$g; rate=$(rate_of "$g")
    if bounded "$cmd_limit" "${C[@]}" new -s "$name" --gpu "$g"; then got=$g; break 2; fi
    maybe_vm=0
    sleep "$retry_sleep" &
    child=$!
    wait "$child"
    child=""
  done
done
[[ -n $got ]] || { stamp "allocation failed on $gpu"; exit 3; }
stamp "allocated gpu=$got"

cat >"$tmp/gpu_check.py" <<'EOF'
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
print("COLAB_JOB_GPU_NAME=" + result.stdout.strip().replace("\n", ";"), flush=True)
EOF
offset=$(wc -c <"$log")
bounded "$check_limit" "${C[@]}" exec -s "$name" -f "$tmp/gpu_check.py" --timeout "$check_limit"
check_rc=$?
gpu_line=$(tail -c +"$((offset + 1))" "$log" | grep -a '^COLAB_JOB_GPU_NAME=' | tail -1)
gpu_name=${gpu_line#COLAB_JOB_GPU_NAME=}
if [[ -z $gpu_name ]] || ! grep -Eq "$(gpu_pattern "$got")" <<<"$gpu_name"; then
  stamp "WRONG GPU: asked for $got, nvidia-smi says '${gpu_name:-nothing}' (check rc=$check_rc)"
  ledger_gpu="$got(got ${gpu_name:-nothing})"
  exit 5
fi
stamp "gpu check passed: $gpu_name"

for p in ${ups[@]+"${ups[@]}"}; do
  stamp "upload ${p%%:*} -> ${p#*:}"
  # Uploads sometimes drop with a broken pipe; retry before giving up.
  ok=""
  for try in 1 2 3; do
    bounded "$cmd_limit" "${C[@]}" upload -s "$name" "${p%%:*}" "${p#*:}" && { ok=1; break; }
    stamp "upload try=$try failed"; sleep "$upload_sleep"
  done
  [[ -n $ok ]] || { stamp "upload failed"; exit 4; }
done

for p in ${pulls[@]+"${pulls[@]}"}; do
  every=${p##*:}; rest=${p%:*}
  stamp "pull every ${every}s ${rest%%:*} -> ${rest#*:}"
  pull_loop "${rest%%:*}" "${rest#*:}" "$every" >>"$log" 2>&1 &
  pull_pids+=("$!")
done

token="COLAB_JOB_OK_$$_$(date +%s)"
run_script="$tmp/$(basename "$script")"
{ cat "$script"; printf "\nprint('\\\\n%s', flush=True)\n" "$token"; } >"$run_script"
stamp "exec $script timeout=${timeout}s"
offset=$(wc -c <"$log")
bounded "$timeout" "${C[@]}" exec -s "$name" -f "$run_script" --timeout "$timeout" ${envs[@]+"${envs[@]}"}
rc=$?
if (( rc == 142 )); then
  stamp "killed at wall-clock limit ${timeout}s"
elif (( rc == 0 )) && ! tail -c +"$((offset + 1))" "$log" | grep -aq "^$token\$"; then
  stamp "the script did not reach its end, read this log"
  rc=6
fi

stop_pulls
for p in ${pulls[@]+"${pulls[@]}"}; do
  rest=${p%:*}
  pull_once "${rest%%:*}" "${rest#*:}" && stamp "last pull ${rest%%:*}" || stamp "last pull of ${rest%%:*} failed, keeping the last copy"
done

download_failed=0
for p in ${downs[@]+"${downs[@]}"}; do
  stamp "download ${p%%:*} -> ${p#*:}"
  ok=""
  for try in 1 2 3; do
    pull_once "${p%%:*}" "${p#*:}" && { ok=1; break; }
    stamp "download try=$try failed"; sleep "$upload_sleep"
  done
  [[ -n $ok ]] || { stamp "download of ${p%%:*} failed"; download_failed=1; }
done
(( rc == 0 && download_failed )) && rc=7

stamp "exec rc=$rc wall_s=$(( $(date +%s) - t0 ))"
exit "$rc"
