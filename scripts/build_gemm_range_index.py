#!/usr/bin/env python3
"""Build optional cold-start indexes for ApxInf GEMM and Attention recipes.

The index contains keys, never trusted configurations. Runtime reopens each
selected hashed .recipe file and validates its full key before use.
"""

import argparse
import os
from pathlib import Path
import tempfile


OPERATORS = {
    "gemm": (("gemm-recipe-v11|",), "gemm-range-index-v1.txt"),
    "attention": (
        ("attention-recipe-v7|", "attention-recipe-v8|"),
        "attention-range-index-v1.txt",
    ),
}


def fnv1a64(key: str) -> int:
    value = 14695981039346656037
    for byte in key.encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def render(directory: Path, operator: str) -> bytes:
    key_prefix, _ = OPERATORS[operator]
    keys = set()
    for path in directory.glob("*.recipe"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 2:
            raise ValueError(f"malformed recipe: {path}")
        key = lines[0]
        if path.name != f"{fnv1a64(key):x}.recipe":
            raise ValueError(f"recipe filename/key mismatch: {path}")
        if not key.startswith(key_prefix):
            continue
        if key in keys:
            raise ValueError(f"duplicate {operator} key: {path}")
        keys.add(key)
    return ("\n".join(sorted(keys)) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--operator", choices=OPERATORS, default="gemm")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"not a recipe directory: {args.directory}")
    output = args.directory / OPERATORS[args.operator][1]
    contents = render(args.directory, args.operator)
    if args.check:
        if not output.is_file() or output.read_bytes() != contents:
            raise SystemExit(f"stale or missing index: {output}")
    else:
        with tempfile.NamedTemporaryFile(dir=args.directory, delete=False) as tmp:
            tmp.write(contents)
            tmp.flush()
            os.fsync(tmp.fileno())
            temporary = Path(tmp.name)
        os.replace(temporary, output)
    print(f"{output}: {contents.count(bytes([10]))} {args.operator} keys")


if __name__ == "__main__":
    main()
