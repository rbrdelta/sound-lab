"""The published records agree with each other, and the repo holds no music or raw readings.

Reads only committed files: runs/*.result.json and *.manifest.json, README.md, the built viewer."""
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# README results table row -> (Qwen2-Audio run, Audio Flamingo run)
TABLE = {"Major vs minor, held chords": ("hearing_controlled", "af_chords"),
         "Major vs minor, 4 s tunes": ("hearing_tunes", "af_tunes"),
         "Roughness": ("qwen_roughness", "af_roughness"),
         "Volume": ("qwen_volume", "af_volume"),
         "Tempo": ("qwen_tempo", "af_tempo")}


def encoder_accuracy(run):
    return json.loads((ROOT / "runs" / f"{run}.result.json").read_text())["layers"]["encoder"]["major_minor"]["accuracy"]


def test_readme_results_table_matches_the_result_files():
    rows = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] in TABLE:
            rows[cells[0]] = cells
    assert set(rows) == set(TABLE)
    for name, (qwen, af) in TABLE.items():
        assert rows[name][2] == f"{round(100 * encoder_accuracy(qwen))}%", name
        assert rows[name][3] == f"{round(100 * encoder_accuracy(af))}%", name


def test_every_run_passed_its_repeat_check():
    for m in sorted((ROOT / "runs").glob("*.manifest.json")):
        assert json.loads(m.read_text())["repeat_check_identical"] is True, m.name


def test_viewer_page_shows_the_result_files_numbers():
    page = (ROOT / "viewer" / "scratch-paper-viewer.html").read_text()
    assert "__DATA__" not in page and "__WALK__" not in page
    data = json.loads(re.search(r"^const DATA = (.*);$", page, re.M).group(1))
    model_runs = {(t, "Qwen2-Audio"): q for t, (q, _) in TABLE.items()} | {(t, "Audio Flamingo"): a for t, (_, a) in TABLE.items()}
    viewer_test = {"Major vs minor — chords": "Major vs minor, held chords", "Major vs minor — tunes": "Major vs minor, 4 s tunes"}
    assert len(data) == 10
    for entry in data:
        run = model_runs[(viewer_test.get(entry["test"], entry["test"]), entry["model"])]
        ears = entry["layers"][0]
        assert ears["name"] == "ears" and ears["acc"] == round(encoder_accuracy(run), 3), run
        assert len(ears["x"]) == len(entry["ids"]) == 144


def test_no_music_or_raw_readings_are_committed():
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    banned = [f for f in files if f.startswith("clips/") or
              Path(f).suffix.lower() in {".npz", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".zip"}]
    assert banned == []
