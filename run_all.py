"""Cross-platform equivalent of `make all` (for machines without make, e.g. Windows)."""

import subprocess
import sys

STEPS = [
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    [sys.executable, "-m", "pytest"],
    [sys.executable, "scripts/run_experiment.py"],
    [sys.executable, "scripts/test_conjecture.py"],
    [sys.executable, "scripts/make_figures.py"],
    [sys.executable, "scripts/update_readme.py"],
]

if __name__ == "__main__":
    for cmd in STEPS:
        print("\n>>", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)
