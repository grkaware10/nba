from __future__ import annotations

import json
from pathlib import Path

from .workflow import outputs_as_dict, run_full_workflow


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    outputs = run_full_workflow(project_root)
    print(json.dumps(outputs_as_dict(outputs), indent=2))


if __name__ == "__main__":
    main()