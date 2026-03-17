import json

from rlm.trajectory import get_trajectory_dir, write_trajectory


def test_write_trajectory_creates_expected_file(tmp_path):
    path = write_trajectory(tmp_path, {"query": "hello", "context_metadata": {"char_count": 5}})

    assert path.exists()
    assert path.parent == get_trajectory_dir(tmp_path)
    assert "_" in path.name
    assert json.loads(path.read_text(encoding="utf-8"))["query"] == "hello"
