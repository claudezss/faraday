from pathlib import Path

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR.parent / "data"

WORKSPACE = ROOT_DIR.parent / "workspace"

WORKSPACE_NETWORKS = WORKSPACE / "networks"

WORKSPACE_NETWORKS.mkdir(parents=True, exist_ok=True)
