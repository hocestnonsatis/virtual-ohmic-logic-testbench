# Project memories

## Stack (2026-07)
- **Language:** Rust (edition 2021), Cargo workspace
- **Crates:** `volt-core` (lib), `volt-cli` (`volt` binary), `volt-py` (PyO3)
- **Build:** `cargo build --release` / `cargo test --workspace`
- **Python:** `maturin develop --release` (PyO3 + abi3-py39); use `.venv` (not `python-source` in pyproject — extension-only)
- **C++/CMake removed** — full rewrite completed 2026-07-06

## CLI
Same flags as before: `--config`, `--weights`, `--inputs`, `--weights2`, `--benchmark`, `--help`

## Tests
- Unit/integration: `crates/volt-core/tests/core.rs`, `equivalence.rs`
- Scenario A MSE regression: `< 1e-6`
