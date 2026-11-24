"""重跑更正到部署：``python -m pairwhite data/capture.json``。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import run_case


def _load(path: Path):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    arrays = {}
    for key, descriptor in envelope["tensors"].items():
        arrays[key] = np.asarray(descriptor["values"], dtype=descriptor["dtype"])
    return envelope["records"], arrays


def main():
    parser = argparse.ArgumentParser(description="重放提交日志，发布分组白化并输出审计报告")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    records, tensors = _load(args.path)
    result = run_case(records, tensors)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
