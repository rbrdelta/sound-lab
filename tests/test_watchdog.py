"""CPU tests for runpod/idle_watchdog.sh with fake nvidia-smi, runpodctl and sleep on PATH.

The fake runpodctl always succeeds and sits first on PATH, so the real one (on a pod) is never
called and the script never reaches its kill-the-container fallback."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "runpod" / "idle_watchdog.sh"


@pytest.fixture
def pod(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (tmp_path / "calls").write_text("")

    def fake(name, body):
        p = bin_ / name
        p.write_text("#!/usr/bin/env bash\n" + body + "\n")
        p.chmod(0o755)

    fake("sleep", "exit 0")  # one check per loop, no waiting
    fake("runpodctl", f'echo "$@" >> {tmp_path}/stopped; exit 0')

    def run(utils, timeout=2, **env):
        """utils: GPU utilisation per check, in order (last value repeats); None = nvidia-smi broken."""
        (tmp_path / "utils").write_text("\n".join("" if u is None else str(u) for u in utils) + "\n")
        fake("nvidia-smi", f'echo x >> {tmp_path}/calls; n=$(wc -l < {tmp_path}/calls); '
                           f'u=$(sed -n "${{n}}p" {tmp_path}/utils); [[ -z "$u" && $n -gt $(wc -l < {tmp_path}/utils) ]] '
                           f'&& u=$(tail -1 {tmp_path}/utils); [[ -z "$u" ]] && exit 1; echo "$u"')
        e = {"PATH": f"{bin_}:{os.environ['PATH']}", "RUNPOD_POD_ID": "test-pod", "GRACE_MINUTES": "0",
             "IDLE_MINUTES": "2", "INTERVAL": "60", "HOLD_FILE": str(tmp_path / "hold"), **env}
        try:
            subprocess.run(["bash", str(SCRIPT)], env=e, timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            pass
        stopped = (tmp_path / "stopped").read_text() if (tmp_path / "stopped").exists() else None
        checks = len((tmp_path / "calls").read_text().splitlines())
        return stopped, checks
    run.dir = tmp_path
    return run


def test_stops_the_pod_after_the_idle_minutes(pod):
    stopped, checks = pod([0])
    assert stopped == "stop pod test-pod\n"
    assert checks == 2  # two idle one-minute checks = IDLE_MINUTES


def test_a_busy_check_restarts_the_idle_count(pod):
    stopped, checks = pod([0, 50, 0, 0])
    assert stopped == "stop pod test-pod\n" and checks == 4


def test_never_stops_while_the_gpu_is_busy(pod):
    stopped, checks = pod([50])
    assert stopped is None and checks > 2


def test_never_stops_blind_when_nvidia_smi_fails(pod):
    stopped, checks = pod([None])
    assert stopped is None and checks > 2


def test_no_stop_inside_the_grace_period(pod):
    stopped, _ = pod([0], GRACE_MINUTES="1")
    assert stopped is None


def test_hold_file_pauses_it(pod):
    (pod.dir / "hold").write_text("")
    stopped, checks = pod([0])
    assert stopped is None and checks == 0
