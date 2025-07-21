from pathlib import Path
from dotenv import load_dotenv
import logging

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR.parent / "data"

WORKSPACE = ROOT_DIR.parent / "workspace"

WORKSPACE_NETWORKS = WORKSPACE / "networks"

WORKSPACE_NETWORKS.mkdir(parents=True, exist_ok=True)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
