#!/usr/bin/env python3
"""Normalize merged Llama config files for SGLang/BFCL compatibility."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging
from scripts.utils.sglang_compat import normalize_model_dir

logger = setup_logging("./logs/make_sglang_compatible.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    logger.info("Normalizing merged model directory: %s", args.model_dir)
    changes = normalize_model_dir(args.model_dir)
    changed = [name for name, did_change in changes.items() if did_change]
    status = ", ".join(changed) if changed else "already compatible"
    logger.info("SGLang/BFCL files normalized: %s", status)


if __name__ == "__main__":
    main()
