#!/usr/bin/env python3
"""Integration test: synthetic GGUF -> CSV -> volt-core loader."""

import os
import subprocess
import sys
import tempfile

try:
    import gguf
except ImportError:
    print("skip: gguf not installed", file=sys.stderr)
    sys.exit(0)

try:
    import volt
except ImportError:
    print("skip: volt not built", file=sys.stderr)
    sys.exit(0)


def write_tiny_gguf(path: str, name: str, data: list[float], shape: list[int]) -> None:
    writer = gguf.GGUFWriter(path, "test")
    import numpy as np

    arr = np.array(data, dtype=np.float32).reshape(shape)
    writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        gguf_path = os.path.join(td, "tiny.gguf")
        csv_path = os.path.join(td, "out.csv")
        write_tiny_gguf(gguf_path, "blk.0.weight", [1.0, 2.0, 3.0, 4.0], [2, 2])

        cmd = [
            sys.executable,
            "python/gguf_to_volt.py",
            "--gguf",
            gguf_path,
            "--tensor",
            "blk.0.weight",
            "--out",
            csv_path,
        ]
        subprocess.check_call(cmd)

        with open(csv_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        assert len(lines) == 2
        row0 = [float(x) for x in lines[0].split(",")]
        assert len(row0) == 2
        assert -1.0 <= row0[0] <= 1.0
        print("gguf_to_volt integration OK")


if __name__ == "__main__":
    main()
