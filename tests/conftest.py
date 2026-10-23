import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def write_run(tmp_path):
    """Write a synthetic run in extract.py's format (<prefix>.npz + <prefix>.manifest.json).

    clips: list of dicts with label, source, recording (clip_id/path filled in). Returns the prefix."""
    def write(encoder, lm, clips, name="run", repeat=True):
        prefix = tmp_path / name
        np.savez_compressed(prefix.with_suffix(".npz"), encoder=np.asarray(encoder), lm=np.asarray(lm))
        man = {"model": "test-model", "prompt": "Listen to this audio clip.", "repeat_check_identical": repeat,
               "clips": [{"clip_id": f"c{i}", "path": f"c{i}.wav", **c} for i, c in enumerate(clips)]}
        prefix.with_suffix(".manifest.json").write_text(json.dumps(man))
        return str(prefix)
    return write
