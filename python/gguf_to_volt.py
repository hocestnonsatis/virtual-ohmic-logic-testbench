#!/usr/bin/env python3
"""Extract a GGUF weight tensor, normalize for VOLT, write CSV."""

from __future__ import annotations

import argparse
import sys

try:
    import gguf
except ImportError as e:
    print("error: pip install gguf", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import numpy as np
except ImportError as e:
    print("error: pip install numpy", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import volt
except ImportError as e:
    print("error: build volt with maturin develop --release", file=sys.stderr)
    raise SystemExit(1) from e


def list_tensors(path: str) -> list[str]:
    reader = gguf.GGUFReader(path)
    return [t.name for t in reader.tensors]


def load_tensor_f64(path: str, name: str) -> tuple[list[float], list[int]]:
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        if tensor.name != name:
            continue
        arr = tensor.data
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        flat = np.asarray(arr, dtype=np.float64).reshape(-1).tolist()
        shape = [int(d) for d in tensor.shape]
        return flat, shape
    raise SystemExit(f"tensor not found: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GGUF tensor -> VOLT weights CSV")
    p.add_argument("--gguf", required=True, help="Path to .gguf file")
    p.add_argument("--tensor", help="Tensor name (e.g. blk.0.attn_q.weight)")
    p.add_argument("--out", help="Output CSV path (required unless --list-tensors)")
    p.add_argument("--row-off", type=int, default=0)
    p.add_argument("--col-off", type=int, default=0)
    p.add_argument("--rows", type=int, default=None, help="Output rows (default: all remaining)")
    p.add_argument("--cols", type=int, default=None, help="Output cols (default: all remaining)")
    p.add_argument("--list-tensors", action="store_true", help="Print tensor names and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tensors:
        for n in list_tensors(args.gguf):
            print(n)
        return

    if not args.out:
        raise SystemExit("error: --out is required unless --list-tensors")
    if not args.tensor:
        raise SystemExit("error: --tensor is required unless --list-tensors")

    flat, shape = load_tensor_f64(args.gguf, args.tensor)
    if len(shape) != 2:
        raise SystemExit(f"expected 2-D tensor, got shape {shape}")

    src_rows, src_cols = shape
    out_rows = args.rows if args.rows is not None else src_rows - args.row_off
    out_cols = args.cols if args.cols is not None else src_cols - args.col_off

    matrix = volt.normalize_weight_matrix(
        flat,
        rows=src_rows,
        cols=src_cols,
        row_off=args.row_off,
        col_off=args.col_off,
        out_rows=out_rows,
        out_cols=out_cols,
    )
    volt.write_weights_csv(args.out, matrix)
    print(f"wrote {out_rows}x{out_cols} weights to {args.out}")


if __name__ == "__main__":
    main()
