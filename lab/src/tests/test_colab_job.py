"""lab/jobs/colab_job.sh against a fake `colab` on PATH. PATH holds the fake and the system directories only, so the real CLI cannot run here."""
import csv
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[2] / "jobs"
WRAPPER = JOBS / "colab_job.sh"
DRIVER = JOBS / "arm-classification.sh"

FAKE = r'''#!{python}
"""A stand-in for the colab CLI. It logs each call, keeps per-command counters, and behaves as $FAKE_CONFIG says.
Like the real `colab exec`, it exits 0 when the executed script raises."""
import json, os, shutil, subprocess, sys, time

args = sys.argv[1:]
base = os.environ["FAKE_DIR"]
config = json.loads(os.environ.get("FAKE_CONFIG", "{{}}"))
with open(os.path.join(base, "calls.jsonl"), "a") as f:
    f.write(json.dumps(args) + "\n")
if args[0] != "--auth=adc":
    print("no auth flag", file=sys.stderr); sys.exit(99)
command = args[1]
counter = os.path.join(base, "count-" + command)
count = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
open(counter, "w").write(str(count))

def option(name):
    return args[args.index(name) + 1]

if command == "new":
    if count <= config.get("new_failures", 0):
        print("capacity refused"); sys.exit(1)
    print("allocated", option("--gpu"))
elif command == "stop":
    if count <= config.get("stop_failures", 0):
        print("stop failed"); sys.exit(1)
    print("stopped")
elif command == "upload":
    if count <= config.get("upload_failures", 0):
        print("broken pipe"); sys.exit(1)
    shutil.copy(args[-2], os.path.join(base, "remote", os.path.basename(args[-1])))
    print("uploaded")
elif command == "download":
    limit = config.get("download_succeeds", None)
    done = os.path.join(base, "count-download-ok")
    ok = int(open(done).read()) if os.path.exists(done) else 0
    source = os.path.join(base, "remote", os.path.basename(args[-2]))
    if count <= config.get("download_failures", 0) or (limit is not None and ok >= limit) or not os.path.exists(source):
        print("download failed"); sys.exit(1)
    shutil.copy(source, args[-1])
    open(done, "w").write(str(ok + 1))
    print("downloaded")
elif command == "exec":
    code = open(option("-f")).read()
    if "COLAB_JOB_GPU_NAME" in code:
        name = config.get("gpu_name", "NVIDIA L4")
        if name is None:
            print("FileNotFoundError: nvidia-smi")
        else:
            print("COLAB_JOB_GPU_NAME=" + name)
        sys.exit(config.get("gpu_check_rc", 0))
    env = dict(os.environ)
    for i, a in enumerate(args):
        if a == "--env":
            key, value = args[i + 1].split("=", 1)
            env[key] = value
    open(os.path.join(base, "exec.pid"), "w").write(str(os.getpid()))
    if config.get("exec_lost_connection"):   # the real client prints this and then hangs with the VM still billing
        print("RuntimeError: Connection was lost.", flush=True)
        time.sleep(600)
    if config.get("exec_sleep"):
        time.sleep(config["exec_sleep"])
    if config.get("exec_rc") is not None:
        sys.exit(config["exec_rc"])
    if config.get("run_script", True):
        subprocess.run([sys.executable, option("-f")], env=env, cwd=os.path.join(base, "remote"))
    else:   # the job itself is not run: only the end token the wrapper appended as the last line
        exec(code.strip().splitlines()[-1])
    sys.exit(0)
else:
    print("unknown command", command); sys.exit(98)
'''


class Harness:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.base = tmp_path / "fake"
        (self.base / "remote").mkdir(parents=True)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        fake = self.bin / "colab"
        fake.write_text(FAKE.format(python=sys.executable))
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.ledger = tmp_path / "results" / "colab-ledger.csv"
        self.log = tmp_path / "logs" / "job.log"
        self.script = tmp_path / "script.py"
        self.script.write_text("print('working')\n")

    def env(self, config=None, **extra):
        path = f"{self.bin}:/usr/bin:/bin"
        assert shutil.which("colab", path=path) == str(self.bin / "colab")
        return {"PATH": path, "HOME": str(self.tmp), "TMPDIR": str(self.tmp), "FAKE_DIR": str(self.base), "FAKE_CONFIG": json.dumps(config or {}), "COLAB_LEDGER": str(self.ledger),
                "COLAB_JOB_RETRY_SLEEP": "0", "COLAB_JOB_UPLOAD_SLEEP": "0", "COLAB_JOB_WATCH_SLEEP": "1", **extra}

    def args(self, *more, charged="sys1"):
        return [str(WRAPPER), "-c", charged, "-n", "job1", "-s", str(self.script), "-l", str(self.log), *[str(m) for m in more]]

    def run(self, *more, config=None, charged="sys1", timeout=60, **extra):
        return subprocess.run(self.args(*more, charged=charged), env=self.env(config, **extra), capture_output=True, text=True, timeout=timeout)

    def start(self, *more, config=None, **extra):
        return subprocess.Popen(self.args(*more), env=self.env(config, **extra), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=lambda: (signal.signal(signal.SIGINT, signal.SIG_DFL), signal.signal(signal.SIGTERM, signal.SIG_DFL)))

    def calls(self):
        path = self.base / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def called(self, command):
        return [c for c in self.calls() if c[1] == command]

    def ledger_rows(self):
        with self.ledger.open() as f:
            return list(csv.DictReader(f))

    def user_exec_started(self):
        return (self.base / "exec.pid").exists()


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def wait_for(condition, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def process_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def assert_released_once(h, name="job1"):
    stops = h.called("stop")
    assert len(stops) == 1, stops
    assert stops[0] == ["--auth=adc", "stop", "-s", name]


def test_success_runs_every_stage_in_order_and_releases(h):
    done = h.run("-u", f"{h.script}:/content/in.py", "-e", "SEED=3", "-t", "600")
    assert done.returncode == 0, done.stdout + done.stderr
    kinds = [c[1] for c in h.calls()]
    assert kinds == ["new", "exec", "upload", "exec", "stop"]
    assert h.called("new")[0] == ["--auth=adc", "new", "-s", "job1", "--gpu", "L4"]
    user = h.called("exec")[1]
    assert user[user.index("--timeout") + 1] == "600" and user[user.index("--env") + 1] == "SEED=3"
    assert_released_once(h)


def test_l4_is_the_default_gpu(h):
    h.run()
    assert h.called("new")[0][-1] == "L4"


def test_all_calls_use_adc_auth(h):
    h.run()
    assert all(c[0] == "--auth=adc" for c in h.calls())


def test_ledger_line_has_the_columns_and_the_l4_rate(h):
    h.script.write_text("import time\ntime.sleep(2.2)\n")
    assert h.run("-t", "60").returncode == 0
    assert h.ledger.read_text().splitlines()[0] == "when,job,gpu,wall_s,cu,charged_to"
    row = h.ledger_rows()[0]
    assert (row["job"], row["gpu"], row["charged_to"]) == ("job1", "L4", "sys1")
    assert int(row["wall_s"]) >= 2
    assert float(row["cu"]) == pytest.approx(int(row["wall_s"]) / 3600 * 1.54, abs=1e-4)
    assert float(row["cu"]) > 0
    assert row["when"].endswith("Z")


def test_ledger_appends_and_charges_whoever_is_named(h):
    h.run()
    h.run(charged="gemma")
    assert [r["charged_to"] for r in h.ledger_rows()] == ["sys1", "gemma"]
    assert len(h.ledger.read_text().splitlines()) == 3


def test_a_higher_rate_applies_when_another_gpu_is_asked_for(h):
    h.script.write_text("import time\ntime.sleep(1.1)\n")
    assert h.run("-g", "A100", config={"gpu_name": "NVIDIA A100-SXM4-40GB"}).returncode == 0
    row = h.ledger_rows()[0]
    assert row["gpu"] == "A100" and float(row["cu"]) == pytest.approx(int(row["wall_s"]) / 3600 * 5.30, abs=1e-4)


@pytest.mark.parametrize("name", ["NVIDIA T4", "Tesla T4", "NVIDIA L40S", "NVIDIA A100-SXM4-40GB", "NVIDIA RTX PRO 6000 Blackwell Server Edition", None, ""])
def test_wrong_gpu_exits_non_zero_runs_nothing_and_releases(h, name):
    done = h.run("-u", f"{h.script}:/content/in.py", "-d", f"/content/out:{h.tmp / 'out'}", config={"gpu_name": name})
    assert done.returncode == 5, done.stdout
    assert not h.user_exec_started() and not h.called("upload") and not h.called("download")
    assert_released_once(h)
    row = h.ledger_rows()[0]
    assert row["gpu"].startswith("L4(got ") and row["charged_to"] == "sys1"


@pytest.mark.parametrize("name", ["NVIDIA L4", "L4", "NVIDIA L4 "])
def test_right_gpu_passes_the_check(h, name):
    assert h.run(config={"gpu_name": name}).returncode == 0
    assert h.user_exec_started()


def test_a_script_that_raises_is_a_failure_even_though_colab_exec_exits_zero(h):
    h.script.write_text("print('start')\nraise RuntimeError('boom')\n")
    done = h.run("-d", f"/content/never:{h.tmp / 'never'}")
    assert done.returncode == 6, done.stdout
    assert_released_once(h)
    assert len(h.ledger_rows()) == 1


def test_the_end_token_cannot_be_forged_by_an_older_log(h):
    h.log.parent.mkdir(parents=True)
    h.log.write_text("COLAB_JOB_OK_1_1\n")
    h.script.write_text("raise RuntimeError('boom')\n")
    assert h.run().returncode == 6


def test_a_non_zero_exec_is_passed_on_and_releases(h):
    done = h.run(config={"exec_rc": 9})
    assert done.returncode == 9
    assert_released_once(h)


def test_outputs_are_still_downloaded_after_a_failed_script(h):
    (h.base / "remote" / "partial.bin").write_text("salvage")
    h.script.write_text("raise RuntimeError('boom')\n")
    assert h.run("-d", f"/content/partial.bin:{h.tmp / 'd' / 'partial.bin'}").returncode == 6
    assert (h.tmp / "d" / "partial.bin").read_text() == "salvage"


def test_allocation_retries_then_succeeds(h):
    assert h.run(config={"new_failures": 2}).returncode == 0
    assert len(h.called("new")) == 3
    assert h.user_exec_started()


def test_allocation_falls_back_along_the_list(h):
    done = h.run("-g", "L4,A100", config={"new_failures": 3, "gpu_name": "NVIDIA A100-SXM4-40GB"})
    assert done.returncode == 0
    assert [c[-1] for c in h.called("new")] == ["L4", "L4", "L4", "A100"]
    assert h.ledger_rows()[0]["gpu"] == "A100"


def test_allocation_failure_exits_3_and_bills_nothing(h):
    done = h.run(config={"new_failures": 99})
    assert done.returncode == 3
    assert len(h.called("new")) == 3
    assert not h.ledger.exists()
    assert_released_once(h)


def test_upload_retries_then_succeeds(h):
    assert h.run("-u", f"{h.script}:/content/in.py", config={"upload_failures": 2}).returncode == 0
    assert len(h.called("upload")) == 3 and (h.base / "remote" / "in.py").exists()


def test_upload_failure_exits_4_and_releases(h):
    done = h.run("-u", f"{h.script}:/content/in.py", config={"upload_failures": 99})
    assert done.returncode == 4
    assert not h.user_exec_started()
    assert_released_once(h)
    assert len(h.ledger_rows()) == 1


def test_download_retries_and_a_lost_download_is_a_failure(h):
    (h.base / "remote" / "out.bin").write_text("x")
    assert h.run("-d", f"/content/out.bin:{h.tmp / 'a.bin'}", config={"download_failures": 2}).returncode == 0
    assert (h.tmp / "a.bin").read_text() == "x"
    (h.base / "remote" / "out.bin").unlink()
    done = h.run("-d", f"/content/out.bin:{h.tmp / 'b.bin'}")
    assert done.returncode == 7
    assert not (h.tmp / "b.bin").exists() and not (h.tmp / "b.bin.part").exists()


def test_a_lost_connection_stops_the_hanging_client_and_releases(h):
    started = time.time()
    done = h.run(config={"exec_lost_connection": True})
    assert done.returncode == 9
    assert time.time() - started < 30
    assert len(h.called("stop")) == 1
    assert len(h.ledger_rows()) == 1


def test_release_retries_and_a_failed_release_is_loud(h):
    assert h.run(config={"stop_failures": 1}).returncode == 0
    assert len(h.called("stop")) == 2
    (h.base / "count-stop").unlink()
    done = h.run(config={"stop_failures": 99})
    assert done.returncode == 8
    assert "A VM MAY STILL BE BILLING" in done.stdout
    assert len(h.ledger_rows()) == 2


@pytest.mark.parametrize("signum,code", [(signal.SIGINT, 130), (signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_a_signal_during_the_script_stops_it_and_releases(h, signum, code):
    process = h.start("-t", "300", config={"exec_sleep": 120})
    assert wait_for(h.user_exec_started)
    time.sleep(0.2)
    pid = int((h.base / "exec.pid").read_text())
    process.send_signal(signum)
    assert process.wait(timeout=20) == code
    assert_released_once(h)
    assert wait_for(lambda: process_gone(pid)), "colab exec still running"
    assert len(h.ledger_rows()) == 1


def test_a_signal_during_allocation_still_releases_and_bills(h):
    (h.bin / "colab").write_text((h.bin / "colab").read_text().replace('if command == "new":', 'if command == "new":\n    time.sleep(120)'))
    process = h.start("-t", "300")
    assert wait_for(lambda: bool(h.called("new")))
    time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=20) == 130
    assert_released_once(h)
    assert len(h.ledger_rows()) == 1


def test_a_second_interrupt_cannot_stop_the_release(h):
    (h.bin / "colab").write_text((h.bin / "colab").read_text().replace('elif command == "stop":', 'elif command == "stop":\n    time.sleep(1.5)'))
    process = h.start("-t", "300", config={"exec_sleep": 120})
    assert wait_for(h.user_exec_started)
    time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    assert wait_for(lambda: bool(h.called("stop")))
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=20) == 130
    assert (h.base / "count-stop").read_text() == "1"
    assert len(h.ledger_rows()) == 1


def test_the_wall_clock_limit_kills_a_hung_script_and_releases(h):
    started = time.time()
    done = h.run("-t", "1", "-d", f"/content/out.bin:{h.tmp / 'out.bin'}", config={"exec_sleep": 120})
    assert done.returncode == 142, done.stdout
    assert time.time() - started < 30
    assert "killed at wall-clock limit" in done.stdout
    assert h.called("download")
    assert_released_once(h)
    assert wait_for(lambda: process_gone(int((h.base / "exec.pid").read_text())))


def test_a_hung_upload_is_cut_by_its_alarm_and_releases(h):
    (h.bin / "colab").write_text((h.bin / "colab").read_text().replace('elif command == "upload":', 'elif command == "upload":\n    time.sleep(120)'))
    started = time.time()
    done = h.run("-u", f"{h.script}:/content/in.py", COLAB_JOB_CMD_LIMIT="1")
    assert done.returncode == 4 and time.time() - started < 30
    assert_released_once(h)


def test_pull_copies_the_checkpoint_while_the_script_runs(h):
    local = h.tmp / "pulled" / "ckpt.tar"
    h.script.write_text(
        "import os, time\n"
        "open('ckpt.tar', 'w').write('v1')\n"
        f"local = {str(local)!r}\n"
        "end = time.time() + 20\n"
        "while not os.path.exists(local) and time.time() < end: time.sleep(0.05)\n"
        "print('SAW_PULL' if os.path.exists(local) else 'NO_PULL', open(local).read() if os.path.exists(local) else '', flush=True)\n"
        "open('ckpt.tar', 'w').write('v2')\n"
    )
    done = h.run("-p", f"/content/ckpt.tar:{local}:1", "-t", "60")
    assert done.returncode == 0, done.stdout
    log = h.log.read_text()
    assert "SAW_PULL v1" in log
    assert local.read_text() == "v2"   # the last pull after the script ended
    assert_released_once(h)


def test_a_failing_pull_keeps_the_last_good_copy_and_does_not_fail_the_job(h):
    local = h.tmp / "pulled" / "ckpt.tar"
    (h.base / "remote" / "ckpt.tar").write_text("good")
    h.script.write_text("import time\ntime.sleep(2.5)\nopen('ckpt.tar', 'w').write('newer')\n")
    done = h.run("-p", f"/content/ckpt.tar:{local}:1", config={"download_succeeds": 1})
    assert done.returncode == 0, done.stdout
    assert local.read_text() == "good"
    assert not local.with_name("ckpt.tar.part").exists()
    assert len(h.called("download")) >= 2


def test_a_pull_of_a_file_that_is_not_there_yet_is_not_a_failure(h):
    assert h.run("-p", f"/content/nothing.tar:{h.tmp / 'n.tar'}:1").returncode == 0
    assert not (h.tmp / "n.tar").exists()


@pytest.mark.parametrize("failure", ["raises", "interrupt", "wrong_gpu"])
def test_no_pull_loop_outlives_the_wrapper(h, failure):
    pull = f"/content/c.tar:{h.tmp / 'c.tar'}:7919"

    def pull_loop_sleeping():
        # A whole-line match, because any shell whose command text merely mentions this sleep would otherwise count.
        listing = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
        return any(line.strip() == "sleep 7919" for line in listing.splitlines())

    if failure == "raises":
        h.script.write_text("raise RuntimeError('boom')\n")
        assert h.run("-p", pull).returncode == 6
    elif failure == "wrong_gpu":
        assert h.run("-p", pull, config={"gpu_name": "NVIDIA T4"}).returncode == 5
    else:
        process = h.start("-p", pull, "-t", "300", config={"exec_sleep": 120})
        assert wait_for(h.user_exec_started)
        assert wait_for(pull_loop_sleeping)   # the check below would pass for nothing if this sleep never showed
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 143
    assert_released_once(h)
    assert wait_for(lambda: not pull_loop_sleeping(), seconds=5)


@pytest.mark.parametrize("args,message", [
    (["-g", "H100"], "no CU rate"),
    (["-t", "28801"], "-t must be"),
    (["-t", "soon"], "-t must be"),
    (["-p", "/content/x:/tmp/x"], "-p needs"),
    (["-p", "/content/x:/tmp/x:0"], "-p needs"),
    (["-u", "/no/such/file:/content/x"], "no such upload"),
])
def test_bad_arguments_exit_2_before_any_vm(h, args, message):
    done = h.run(*args)
    assert done.returncode == 2 and message in done.stderr
    assert not h.calls()


@pytest.mark.parametrize("drop", ["-c", "-n", "-s", "-l"])
def test_required_options_exit_2_before_any_vm(h, drop):
    full = {"-c": "sys1", "-n": "job1", "-s": str(h.script), "-l": str(h.log)}
    del full[drop]
    done = subprocess.run([str(WRAPPER), *[x for pair in full.items() for x in pair]], env=h.env(), capture_output=True, text=True)
    assert done.returncode == 2
    assert not h.calls()


def test_charging_nobody_known_is_refused(h):
    done = h.run(charged="halil")
    assert done.returncode == 2 and not h.calls()


def test_ledger_reads_as_check_gates_reads_it(h):
    h.run()
    h.run(charged="gemma")
    sys1 = gemma = 0.0
    for row in h.ledger_rows():
        if row["charged_to"] == "gemma":
            gemma += float(row["cu"])
        else:
            sys1 += float(row["cu"])
    assert sys1 >= 0 and gemma >= 0


# The Mac-side driver of arm-classification.


def driver_inputs(h, tokens=None):
    data = h.tmp / "data"
    data.mkdir()
    (data / "train.jsonl.gz").write_bytes(b"x")
    (data / "dev.jsonl.gz").write_bytes(b"x")
    (data / "train.report.json").write_text(json.dumps({"tokens": {"max_state_needed": 612}} if tokens is None else tokens))
    python = Path(sys.executable)
    return {"TRAIN_JSONL": str(data / "train.jsonl.gz"), "DEV_JSONL": str(data / "dev.jsonl.gz"), "RESULTS_DIR": str(h.tmp / "results" / "arm-classification"), "PACK_DIR": str(h.tmp / "pack"), "PYTHON": str(python)}


def run_driver(h, *args, config=None, **extra):
    env = {**h.env({"run_script": False, **(config or {})}), **driver_inputs(h), **extra}
    return subprocess.run([str(DRIVER), *args], env=env, capture_output=True, text=True, timeout=60)


def test_driver_speedtest_uploads_inputs_passes_settings_and_downloads_away_from_trained_results(h):
    (h.base / "remote" / "seed4.json").write_text("{}")
    done = run_driver(h, "4", "speedtest", BATCH="4")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (h.tmp / "results" / "arm-classification" / "speedtest" / "seed4.json").exists()
    assert not (h.tmp / "results" / "arm-classification" / "seed4.json").exists()
    uploads = {c[-1]: c[-2] for c in h.called("upload")}
    assert set(uploads) == {"/content/train.jsonl.gz", "/content/dev.jsonl.gz", "/content/sys1_code.tar.gz"}
    user = h.called("exec")[1]
    sent = {user[i + 1].split("=")[0]: user[i + 1].split("=", 1)[1] for i, a in enumerate(user) if a == "--env"}
    assert sent["SEED"] == "4" and sent["MODE"] == "speedtest" and sent["MAX_STATE"] == "612" and sent["BATCH"] == "4"
    assert "GIT_COMMIT" in sent and "GIT_DIRTY" in sent and "EPOCHS" not in sent
    download = h.called("download")[0]
    assert download[-2] == "/content/out/seed4.json" and "/speedtest/" in download[-1]
    assert h.called("new")[0][-1] == "L4"
    assert h.ledger_rows()[0]["charged_to"] == "sys1"
    assert_released_once(h, "arm-classification-seed4-speedtest")


def test_driver_full_run_pulls_partial_weights_into_the_ignored_folder(h):
    for name in ("seed4.json", "seed4.tar", "seed4-partial.tar"):
        (h.base / "remote" / name).write_text(name)
    done = run_driver(h, "4", "full", ALLOW_DIRTY="1")
    assert done.returncode == 0, done.stdout + done.stderr
    results = h.tmp / "results" / "arm-classification"
    assert (results / "seed4.json").read_text() == "seed4.json"
    assert (results / "weights" / "seed4.tar").read_text() == "seed4.tar"
    assert (results / "weights" / "seed4-partial.tar").read_text() == "seed4-partial.tar"
    assert "/content/out/seed4-partial.tar" in [c[-2] for c in h.called("download")]


def test_driver_full_run_without_a_result_file_is_a_failure(h):
    assert run_driver(h, "4", "full", ALLOW_DIRTY="1").returncode == 7
    assert_released_once(h, "arm-classification-seed4")


def test_driver_refuses_a_full_run_on_uncommitted_code_unless_told(h):
    probe = subprocess.run(["git", "-C", str(JOBS), "status", "--porcelain", "--", "lab/jobs"], capture_output=True, text=True)
    done = run_driver(h, "4", "full")
    if probe.stdout.strip() or subprocess.run(["git", "-C", str(JOBS.parents[1]), "status", "--porcelain", "--", "lab/src/sys1", "lab/src/train", "lab/jobs"], capture_output=True, text=True).stdout.strip():
        assert done.returncode == 2 and "commit first" in done.stderr
        assert not h.calls()
    else:
        assert done.returncode == 7   # clean tree: it runs, and the fake has no result file to fetch


def test_driver_refuses_a_report_without_tokens(h):
    env = {**h.env({"run_script": False}), **driver_inputs(h, tokens={"counts": {}})}
    done = subprocess.run([str(DRIVER), "4", "speedtest"], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and "tokens.max_state_needed" in done.stderr
    assert not h.calls()


@pytest.mark.parametrize("args", [["x"], ["4", "other"], []])
def test_driver_refuses_bad_arguments(h, args):
    done = subprocess.run([str(DRIVER), *args], env=h.env(), capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and not h.calls()
