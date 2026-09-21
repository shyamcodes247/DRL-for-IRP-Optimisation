import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
AGENT_DIR = SRC_DIR / "agent"
for _p in (SRC_DIR, AGENT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = PROJECT_ROOT / "data" / "Instances_lowcost_H6"


def _read_manifest(filename: str) -> List[str]:
    """
    Reads a manifest file (a plain list of instance filenames, one per line —
    e.g. `test_instances.dat`, `comp_format_instances.dat`) and resolves each
    name to a full path under `DATA_DIR`.
    """
    manifest_path = DATA_DIR / filename
    with open(manifest_path) as f:
        names = [line.strip() for line in f if line.strip()]
    return [str(DATA_DIR / name) for name in names]


# `test_instances.dat` names the (small) set of instances fast/targeted tests
# should run against; `comp_format_instances.dat` lists the full benchmark
# suite, used only for cheap checks (e.g. that every instance file parses).
TEST_INSTANCE_PATHS = _read_manifest("test_instances.dat")
ALL_INSTANCE_PATHS = _read_manifest("comp_format_instances.dat")
