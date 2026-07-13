# GGUF Weight Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a named weight tensor from a GGUF model file, normalize it to VOLT's `[-1, 1]` range, optionally slice to ≤512×512, write CSV, and run the existing `volt --weights` pipeline.

**Architecture:** Keep `volt-core` dependency-light. Add pure-Rust matrix normalization/slicing in `weight_norm.rs` and CSV export in `weights_csv.rs`. Add a Python CLI (`python/gguf_to_volt.py`) that uses the community `gguf` package to read tensors and calls new PyO3 helpers on the existing `volt` module for normalization and CSV writing. No full LLM inference — only single-tensor extraction for crossbar simulation.

**Tech Stack:** Rust 2021 (volt-core, volt-py), PyO3 0.29+, Python 3.9+, `gguf` pip package, existing CSV/CLI paths.

**Spec note:** No brainstorming spec document was found in the repo. This plan targets the natural Phase 6 feature discussed in prior sessions (GGUF → VOLT weight pipeline). If a different feature was intended, replace this plan before execution.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `crates/volt-core/src/weight_norm.rs` | Reshape flat tensors, slice submatrix, symmetric min-max normalize to `[-1, 1]`, enforce 512×512 limits |
| `crates/volt-core/src/weights_csv.rs` | Add `write_weights_csv_file` (mirror of existing loader) |
| `crates/volt-core/src/lib.rs` | Export new public API |
| `crates/volt-core/tests/core.rs` | Unit tests for norm + CSV round-trip |
| `crates/volt-py/src/lib.rs` | PyO3 bindings: `normalize_weight_matrix`, `write_weights_csv` |
| `python/gguf_to_volt.py` | CLI: `--gguf`, `--tensor`, slice flags, `--out` |
| `python/test_gguf_to_volt.py` | Generates tiny synthetic GGUF fixture, runs extractor |
| `README.md` | "Importing weights from GGUF" section |
| `memories.md` | Note GGUF workflow and Python dep |

Files that do **not** change in v1 (YAGNI): `volt-cli` (user runs extractor then `volt --weights`), no native Rust GGUF parser crate.

---

### Task 1: Weight normalization module

**Files:**
- Create: `crates/volt-core/src/weight_norm.rs`
- Modify: `crates/volt-core/src/lib.rs`
- Test: `crates/volt-core/tests/core.rs`

- [ ] **Step 1: Write the failing test**

Add to `crates/volt-core/tests/core.rs`:

```rust
use volt_core::{
    extract_submatrix, normalize_to_symmetric_range, reshape_row_major,
    write_weights_csv_file, load_weights_csv_file,
};

#[test]
fn weight_norm_symmetric_range() {
    let raw = vec![-10.0, 0.0, 10.0];
    let norm = normalize_to_symmetric_range(&raw).unwrap();
    assert_near(norm[0] as f32, -1.0, 1e-12, "min -> -1");
    assert_near(norm[1] as f32, 0.0, 1e-12, "mid -> 0");
    assert_near(norm[2] as f32, 1.0, 1e-12, "max -> 1");
}

#[test]
fn weight_norm_flat_constant_is_zero() {
    let raw = vec![3.0, 3.0, 3.0];
    let norm = normalize_to_symmetric_range(&raw).unwrap();
    assert_near(norm[0] as f32, 0.0, 1e-12, "constant -> 0");
    assert_near(norm[2] as f32, 0.0, 1e-12, "constant -> 0");
}

#[test]
fn weight_norm_reshape_and_slice() {
    let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let full = reshape_row_major(&flat, 2, 3).unwrap();
    let sub = extract_submatrix(&full, 0, 1, 2, 2).unwrap();
    assert_eq!(sub.len(), 2);
    assert_eq!(sub[0], vec![2.0, 3.0]);
    assert_eq!(sub[1], vec![5.0, 6.0]);
}
```

Also add `write_weights_csv_file` and `load_weights_csv_file` to the existing `use volt_core::{...}` import at the top of `core.rs` (merge with current imports).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p volt-core weight_norm -- --nocapture`

Expected: FAIL with `cannot find function normalize_to_symmetric_range` (or module not found).

- [ ] **Step 3: Write minimal implementation**

Create `crates/volt-core/src/weight_norm.rs`:

```rust
use crate::weights_csv::{K_MAX_WEIGHTS_COLS, K_MAX_WEIGHTS_ROWS};

pub fn normalize_to_symmetric_range(values: &[f64]) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return Err("normalize_to_symmetric_range: empty input".into());
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let span = max_v - min_v;
    if span <= 0.0 {
        return Ok(vec![0.0; values.len()]);
    }
    Ok(values
        .iter()
        .map(|&v| 2.0 * (v - min_v) / span - 1.0)
        .collect())
}

pub fn reshape_row_major(flat: &[f64], rows: usize, cols: usize) -> Result<Vec<Vec<f64>>, String> {
    let need = rows.checked_mul(cols).ok_or_else(|| "reshape_row_major: overflow".to_string())?;
    if flat.len() != need {
        return Err(format!(
            "reshape_row_major: expected {} elements, got {}",
            need,
            flat.len()
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        out.push(flat[start..start + cols].to_vec());
    }
    Ok(out)
}

pub fn extract_submatrix(
    matrix: &[Vec<f64>],
    row_off: usize,
    col_off: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if matrix.is_empty() {
        return Err("extract_submatrix: empty matrix".into());
    }
    let src_cols = matrix[0].len();
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != src_cols {
            return Err(format!("extract_submatrix: ragged row {i}"));
        }
    }
    if row_off + n_rows > matrix.len() {
        return Err("extract_submatrix: row slice out of bounds".into());
    }
    if col_off + n_cols > src_cols {
        return Err("extract_submatrix: column slice out of bounds".into());
    }
    if n_rows > K_MAX_WEIGHTS_ROWS as usize {
        return Err("extract_submatrix: row count exceeds k_max_weights_rows".into());
    }
    if n_cols > K_MAX_WEIGHTS_COLS as usize {
        return Err("extract_submatrix: column count exceeds k_max_weights_cols".into());
    }
    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        out.push(matrix[row_off + r][col_off..col_off + n_cols].to_vec());
    }
    Ok(out)
}

pub fn prepare_weight_matrix(
    flat: &[f64],
    rows: usize,
    cols: usize,
    row_off: usize,
    col_off: usize,
    out_rows: usize,
    out_cols: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let full = reshape_row_major(flat, rows, cols)?;
    let sliced = extract_submatrix(&full, row_off, col_off, out_rows, out_cols)?;
    let mut normalized_rows = Vec::with_capacity(out_rows);
    for row in sliced {
        let norm_row = normalize_to_symmetric_range(&row)?;
        normalized_rows.push(norm_row);
    }
    Ok(normalized_rows)
}
```

Add to `crates/volt-core/src/lib.rs`:

```rust
pub mod weight_norm;

pub use weight_norm::{
    extract_submatrix, normalize_to_symmetric_range, prepare_weight_matrix, reshape_row_major,
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p volt-core weight_norm -- --nocapture`

Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add crates/volt-core/src/weight_norm.rs crates/volt-core/src/lib.rs crates/volt-core/tests/core.rs
git commit -m "feat(core): add weight normalization and submatrix slicing"
```

---

### Task 2: CSV weight writer

**Files:**
- Modify: `crates/volt-core/src/weights_csv.rs`
- Modify: `crates/volt-core/src/lib.rs`
- Test: `crates/volt-core/tests/core.rs`

- [ ] **Step 1: Write the failing test**

Add to `crates/volt-core/tests/core.rs`:

```rust
#[test]
fn weights_csv_round_trip() {
    let path = "volt_test_weights_roundtrip.csv";
    let original = vec![vec![0.8, -0.3], vec![-0.6, 0.9]];
    write_weights_csv_file(path, &original).unwrap();
    let loaded = load_weights_csv_file(path).unwrap();
    let _ = fs::remove_file(path);
    assert_eq!(loaded.len(), 2);
    assert_near(loaded[0][0] as f32, 0.8, 1e-9, "rt w00");
    assert_near(loaded[1][1] as f32, 0.9, 1e-9, "rt w11");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p volt-core weights_csv_round_trip -- --nocapture`

Expected: FAIL with `cannot find function write_weights_csv_file`.

- [ ] **Step 3: Write minimal implementation**

Append to `crates/volt-core/src/weights_csv.rs`:

```rust
use std::io::Write;

pub fn write_weights_csv_file(path: &str, weights: &[Vec<f64>]) -> Result<(), String> {
    if weights.is_empty() {
        return Err("write_weights_csv: empty matrix".into());
    }
    let cols = weights[0].len();
    if cols == 0 {
        return Err("write_weights_csv: empty row".into());
    }
    for (i, row) in weights.iter().enumerate() {
        if row.len() != cols {
            return Err(format!("write_weights_csv: ragged row {i}"));
        }
    }
    if weights.len() > K_MAX_WEIGHTS_ROWS as usize {
        return Err("write_weights_csv: row count exceeds k_max_weights_rows".into());
    }
    if cols > K_MAX_WEIGHTS_COLS as usize {
        return Err("write_weights_csv: column count exceeds k_max_weights_cols".into());
    }
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("write_weights_csv: cannot create {path}: {e}"))?;
    writeln!(f, "# VOLT weights export")
        .map_err(|e| format!("write_weights_csv: write error: {e}"))?;
    for row in weights {
        let line: Vec<String> = row.iter().map(|v| format!("{v:.17}")).collect();
        writeln!(f, "{}", line.join(","))
            .map_err(|e| format!("write_weights_csv: write error: {e}"))?;
    }
    Ok(())
}
```

Update `crates/volt-core/src/lib.rs` export:

```rust
pub use weights_csv::{
    load_inputs_csv_file, load_weights_csv_file, write_weights_csv_file, K_MAX_WEIGHTS_COLS,
    K_MAX_WEIGHTS_ROWS,
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p volt-core weights_csv_round_trip -- --nocapture`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/volt-core/src/weights_csv.rs crates/volt-core/src/lib.rs crates/volt-core/tests/core.rs
git commit -m "feat(core): add CSV weight export for GGUF pipeline"
```

---

### Task 3: PyO3 bindings for Python extractor

**Files:**
- Modify: `crates/volt-py/src/lib.rs`
- Test: `python/test_gguf_to_volt.py` (added in Task 4; for this task use `python/smoke_test.py` extension)

- [ ] **Step 1: Write the failing test**

Extend `python/smoke_test.py`:

```python
flat = [1.0, 2.0, 3.0, 4.0]
matrix = volt.normalize_weight_matrix(flat, rows=2, cols=2, row_off=0, col_off=0, out_rows=2, out_cols=2)
assert len(matrix) == 2
assert len(matrix[0]) == 2
volt.write_weights_csv("volt_smoke_export.csv", matrix)
import os
assert os.path.isfile("volt_smoke_export.csv")
os.remove("volt_smoke_export.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `maturin develop --release -C crates/volt-py && python python/smoke_test.py`

Expected: FAIL with `AttributeError: module 'volt' has no attribute 'normalize_weight_matrix'`

- [ ] **Step 3: Write minimal implementation**

Add to `crates/volt-py/src/lib.rs` imports:

```rust
use volt_core::{prepare_weight_matrix, write_weights_csv_file};
```

Add functions before `#[pymodule]`:

```rust
#[pyfunction]
#[pyo3(signature = (flat, rows, cols, row_off=0, col_off=0, out_rows=None, out_cols=None))]
fn normalize_weight_matrix(
    flat: Vec<f64>,
    rows: usize,
    cols: usize,
    row_off: usize,
    col_off: usize,
    out_rows: Option<usize>,
    out_cols: Option<usize>,
) -> PyResult<Vec<Vec<f32>>> {
    let or = out_rows.unwrap_or(rows.saturating_sub(row_off));
    let oc = out_cols.unwrap_or(cols.saturating_sub(col_off));
    let m = prepare_weight_matrix(&flat, rows, cols, row_off, col_off, or, oc)
        .map_err(PyRuntimeError::new_err)?;
    Ok(m.into_iter()
        .map(|row| row.into_iter().map(|v| v as f32).collect())
        .collect())
}

#[pyfunction]
fn write_weights_csv(path: &str, weights: Vec<Vec<f32>>) -> PyResult<()> {
    let wd: Vec<Vec<f64>> = weights
        .iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect();
    write_weights_csv_file(path, &wd).map_err(PyRuntimeError::new_err)
}
```

Register in `volt` module:

```rust
m.add_function(wrap_pyfunction!(normalize_weight_matrix, m)?)?;
m.add_function(wrap_pyfunction!(write_weights_csv, m)?)?;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `maturin develop --release -C crates/volt-py && python python/smoke_test.py`

Expected: `python smoke test OK`

- [ ] **Step 5: Commit**

```bash
git add crates/volt-py/src/lib.rs python/smoke_test.py
git commit -m "feat(py): expose weight normalize and CSV write for GGUF import"
```

---

### Task 4: Python GGUF extractor CLI

**Files:**
- Create: `python/gguf_to_volt.py`
- Create: `python/test_gguf_to_volt.py`
- Modify: `.github/workflows/python-bindings.yml`

- [ ] **Step 1: Write the failing test**

Create `python/test_gguf_to_volt.py`:

```python
#!/usr/bin/env python3
"""Integration test: synthetic GGUF -> CSV -> volt-core loader."""

import os
import struct
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install gguf numpy && maturin develop --release -C crates/volt-py && python python/test_gguf_to_volt.py`

Expected: FAIL — `python/gguf_to_volt.py` not found or script errors.

- [ ] **Step 3: Write minimal implementation**

Create `python/gguf_to_volt.py`:

```python
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
```

Add CI step to `.github/workflows/python-bindings.yml` (after maturin build step):

```yaml
      - name: Install gguf for import test
        run: pip install gguf numpy

      - name: GGUF import integration test
        run: python python/test_gguf_to_volt.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install gguf numpy && maturin develop --release -C crates/volt-py && python python/test_gguf_to_volt.py`

Expected: `gguf_to_volt integration OK`

- [ ] **Step 5: Commit**

```bash
git add python/gguf_to_volt.py python/test_gguf_to_volt.py .github/workflows/python-bindings.yml
git commit -m "feat(python): add GGUF tensor to VOLT CSV extractor"
```

---

### Task 5: End-to-end CLI smoke

**Files:**
- Test: manual + CI (existing workflow)

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`

Expected: all tests PASS (including Scenario A MSE `< 1e-6`).

- [ ] **Step 2: Run GGUF → CSV → volt pipeline**

Run:

```bash
python python/test_gguf_to_volt.py
OUT=$(mktemp).csv
python python/gguf_to_volt.py --gguf /path/to/tiny.gguf --tensor blk.0.weight --out "$OUT"  # use fixture from test
cargo run --release -p volt-cli -- --weights "$OUT" --inputs volt.example.inputs.csv
rm -f "$OUT"
```

For the manual step, reuse the synthetic GGUF from `test_gguf_to_volt.py` or any small local `.gguf` file.

Expected: `volt` exits 0 and writes `results.csv`.

- [ ] **Step 3: Commit** (only if doc changes in Task 6 were batched; otherwise skip empty commit)

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`
- Modify: `memories.md`

- [ ] **Step 1: Add README section**

Insert after the CSV weight import bullet in README (Roadmap or Quick start area):

```markdown
### Importing weights from GGUF

VOLT does not run full LLM inference. To simulate one layer from a GGUF checkpoint:

1. Build Python bindings: `maturin develop --release`
2. Install helper: `pip install gguf numpy`
3. List tensors: `python python/gguf_to_volt.py --gguf model.gguf --list-tensors`
4. Extract slice (max 512×512): `python python/gguf_to_volt.py --gguf model.gguf --tensor blk.0.attn_q.weight --rows 128 --cols 128 --out layer0.csv`
5. Run simulation: `./volt --weights layer0.csv --inputs volt.example.inputs.csv`

Weights are min-max normalized to `[-1, 1]` per row slice before export.
```

- [ ] **Step 2: Update memories.md**

Add under project status:

```markdown
- **GGUF import:** `python/gguf_to_volt.py` + PyO3 `normalize_weight_matrix` / `write_weights_csv`; requires `pip install gguf numpy`; max 512×512 per layer.
```

- [ ] **Step 3: Commit**

```bash
git add README.md memories.md
git commit -m "docs: document GGUF to VOLT weight import workflow"
```

---

## Self-Review

**1. Spec coverage**

| Requirement | Task |
|-------------|------|
| Read GGUF tensor by name | Task 4 (`gguf_to_volt.py`) |
| Normalize to `[-1, 1]` | Task 1 (`normalize_to_symmetric_range`, per-row in `prepare_weight_matrix`) |
| Respect 512×512 limit | Task 1 (`extract_submatrix` checks `K_MAX_*`) |
| Export CSV for existing CLI | Task 2 + Task 4 |
| Python + Rust single source of truth for norm | Task 1 + Task 3 |
| CI coverage | Task 4 (workflow step) |
| User documentation | Task 6 |

Gap (explicitly out of scope v1): full model graph, tokenizer, multi-layer auto-chain from GGUF metadata, native Rust GGUF parser.

**2. Placeholder scan:** No TBD/TODO/similar-to placeholders found.

**3. Type consistency:** `prepare_weight_matrix` returns `Vec<Vec<f64>>`; PyO3 converts to `Vec<Vec<f32>>` for Python; CSV writer accepts `f64`; loader returns `f64` — consistent with existing `weights_csv` and `CrossbarArray::load_weights` (`f32` cast at load site).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-13-gguf-weight-import.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
