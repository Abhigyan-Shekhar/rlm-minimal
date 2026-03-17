import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any


def get_trajectory_dir(state_dir: Path) -> Path:
    return state_dir / "trajectories"


def write_trajectory(state_dir: Path, payload: dict[str, Any]) -> Path:
    trajectory_dir = get_trajectory_dir(state_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    short_id = secrets.token_hex(3)
    path = trajectory_dir / f"{timestamp}_{short_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
