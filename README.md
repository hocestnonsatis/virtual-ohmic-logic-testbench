# VOLT — Virtual Ohmic Logic Testbench

<p align="center">
  <a href="https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/releases/latest"><img src="https://img.shields.io/github/v/release/hocestnonsatis/virtual-ohmic-logic-testbench?sort=semver&amp;logo=github&amp;label=release" alt="Latest release" /></a>
  &nbsp;
  <a href="https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/actions/workflows/cmake-multi-platform.yml"><img src="https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/actions/workflows/cmake-multi-platform.yml/badge.svg" alt="CI" /></a>
  &nbsp;
  <a href="https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/blob/main/LICENSE"><img src="https://img.shields.io/github/license/hocestnonsatis/virtual-ohmic-logic-testbench?label=license" alt="License" /></a>
  &nbsp;
  <img src="https://img.shields.io/badge/prebuilt-Linux%20%7C%20Windows%20%7C%20macOS-2ea043?logo=rocket&logoColor=white" alt="Platforms" />
</p>

<p align="center">
  <sub>Tree version <code>0.0.1</code> · Tag a release as <code>v0.0.1</code> to publish binaries via GitHub Actions · <a href="#releases--downloads">Downloads</a></sub>
</p>

Software physics simulator for **Analog In-Memory Computing (AIMC)** crossbar arrays. VOLT models voltage-based neural inference with Ohm’s law and Kirchhoff’s current law—useful for exploring feasibility before hardware exists.

**Stack:** C++17, CMake ≥ 3.14, standard library only (no third-party dependencies).

---

## Contents

1. [Concept](#concept) — digital operations vs. simulated analog  
2. [Quick start](#quick-start) — build, run, tests  
3. [Releases & downloads](#releases--downloads) — prebuilt `volt`, versioning  
4. [Repository layout](#repository-layout)  
5. [Design notes](#design-notes) — differential pair, bipolar ADC, read disturb, activations, endurance  
6. [Scenarios & results](#scenarios--results)  
7. [Defaults (`config.hpp`)](#defaults-confighpp)  
8. [Project rules](#project-rules)  
9. [Roadmap](#roadmap)

---

## Concept

| Digital idea | Analog stand-in | Formula |
|--------------|-----------------|---------|
| Input token | Voltage `V` | `V = V_min + input × (V_max − V_min)` |
| Weight | Conductance `G` | `G = f(w)` via differential pair |
| Multiply | Ohm’s law | `I = V × G` |
| Dot product | Kirchhoff (current sum) | `I_total = Σ (Vᵢ × Gᵢ)` |
| Output | ADC | `level = floor((I − I_min) / I_step)` |

---

## Quick start

**Requirements:** CMake ≥ 3.14; GCC or Clang with C++17.

```bash
cmake -S . -B build
cmake --build build
```

Run the simulator from the build directory (so `results.csv` is written next to the binary):

```bash
cd build && ./volt
```

**Benchmark mode** (matrix size sweep, ~40 ms per size; writes `benchmark.csv`):

```bash
cd build && ./volt --benchmark
```

**Physics JSON** (optional; merged onto defaults for all scenarios and for `--benchmark`):

```bash
cd build && ./volt --config ../volt.example.json
```

**Weight matrix CSV** (optional square **N×N**, N ≤ 512; comma-separated rows):

```bash
cd build && ./volt --weights ../volt.example.weights.csv
```

**Inputs CSV** (optional; **N** values in `[0,1]` for the DAC — one line and/or one number per line). If you omit `--inputs` after `--weights`, a built-in ramp vector is used; with the built-in 4×4 demo (no `--weights`), the original demo input is kept.

```bash
cd build && ./volt --weights ../volt.example.weights.csv --inputs ../volt.example.inputs.csv
```

**Second layer (scenario F)** — optional **`--weights2`** N×N CSV; default is **0.5×I** so currents stay easier to bound.

Run tests:

```bash
cd build && ctest --output-on-failure
```

---

## Releases & downloads

| | |
|:--|:--|
| **Latest** | [![GitHub release](https://img.shields.io/github/v/release/hocestnonsatis/virtual-ohmic-logic-testbench?logo=github)](https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/releases/latest) |
| **Version file** | [`VERSION`](VERSION) (current `0.0.1`) |

Pushing a git tag matching `v*.*.*` (for example `v0.0.1`) runs [`.github/workflows/release.yml`](.github/workflows/release.yml): **Linux** (g++, `.tar.gz`), **Windows** (MSVC, `.zip`), **macOS** (Clang, `.tar.gz`). Each archive contains the `volt` binary (`volt.exe` on Windows), `README.md`, `VERSION`, and `LICENSE`.

| Platform | Asset name pattern |
|----------|-------------------|
| Linux x86_64 | `volt-<tag>-linux-x86_64.tar.gz` |
| Windows x64 | `volt-<tag>-windows-x64.zip` |
| macOS | `volt-<tag>-macos.tar.gz` |

After extracting, run `./volt` from the inner folder (Linux/macOS) or `volt.exe` (Windows). CI status for every push/PR: [![CI](https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/actions/workflows/cmake-multi-platform.yml).

---

## Repository layout

```
.
├── VERSION                 # Semver string for releases (e.g. 0.0.1)
├── LICENSE                 # MIT
├── volt.example.json       # Example `--config` (subset of fields)
├── volt.example.weights.csv # Example `--weights` (4×4)
├── volt.example.inputs.csv  # Example `--inputs` (length 4)
├── src/
│   ├── config.hpp          # Physical constants
│   ├── config_json.hpp / .cpp # Optional JSON overlay for physics params
│   ├── weights_csv.hpp / .cpp # Optional `--weights` CSV import
│   ├── dac.hpp / dac.cpp   # Digital → voltage
│   ├── adc.hpp / adc.cpp   # Current → digital
│   ├── crossbar.hpp / .cpp # Weight matrix (differential pair)
│   ├── noise.hpp / .cpp    # Thermal noise, read disturb, write endurance
│   ├── activation.hpp / .cpp # ReLU / sigmoid on I_net (optional)
│   ├── benchmark.hpp / .cpp  # Optional `--benchmark` sweep
│   └── main.cpp            # Pipeline + scenarios A–I
├── tests/
│   ├── test_core.cpp
│   └── test_equivalence.cpp  # Regression: MSE < 1e-6 (ideal path)
├── .github/workflows/
│   ├── cmake-multi-platform.yml
│   └── release.yml         # Tag v*.*.* → Release + Linux / Windows / macOS binaries
└── CMakeLists.txt
```

---

## Design notes

### Differential pair (signed weights)

Conductance is physical: `G ≥ 0`. Signed weights use two cells per effective weight:

```
G_pos[i][j] = (w + 1) / 2 × G_max
G_neg[i][j] = (1 − w) / 2 × G_max
I_net       = (G_pos − G_neg) × V = w × G_max × V
```

This keeps weights in `[-1, 1]` while respecting `G ≥ 0`.

### Bipolar ADC window

`I_net` can be negative. The ADC maps the full signed current range `[I_min, I_min + I_range]` to digital codes:

```
I_step = I_range / (2^n_bits − 1)
level  = clamp(floor((I − I_min) / I_step), 0, 2^n_bits − 1)
```

A unipolar-only window collapses quantization and hurts SNR—a common AIMC pitfall.

### RRAM read disturb

Each read perturbs neighbors in proportion to applied voltage:

```
V_dis = V_applied × disturb_ratio   (default 3%)
δG    = alpha × V_dis               (default alpha: 1e−5)
```

Only adjacent rows (±1) are updated per read.

### Analog activation (optional)

After the MAC (`I_net` per column), an optional **ReLU** or **sigmoid** maps current before the ADC. ReLU is `max(0, I)`. Sigmoid uses the window midpoint and span from `config.hpp`:

```
mid = I_min + I_range / 2
x   = ((I − mid) / (I_range / 4)) × activation_sigmoid_steepness
I′  = I_min + I_range × σ(x)
```

Reference currents apply the same nonlinearity to the ideal linear `I_net` so MSE/SNR stay meaningful.

### Write endurance (optional)

After programming, **cumulative write/erase stress** is modeled as a uniform scale on all conductances:

```
G_pos, G_neg ← clamp(s × G_pos, s × G_neg)   with   s = exp(−write_endurance_lambda × cycles)
G_max effective ← G_max × s
```

Reference currents use `CrossbarArray::effective_g_max()` so MSE compares to the same weakened linear model. Read disturb and thermal noise clamps use the current effective ceiling.

### Benchmark mode

`./volt --benchmark` times steady-state `CrossbarArray::apply_voltage` for n × n arrays with deterministic weights and DAC inputs. Each sweep step runs for about 40 ms wall time; **GMAC/s** uses n² MACs per forward (one multiply per matrix entry contributing to each output column). Output: `benchmark.csv` plus a short human-readable summary on stdout.

---

## Scenarios & results

Nine scenarios use one **N×N** weight matrix (default **N** = 4) and an **N**-vector input. Scenario **F** uses a second **N×N** matrix (default **0.5×I**). Output is **`results.csv`** in the working directory when you run `./volt` (typically `build/results.csv`). The CSV includes **`endurance_cycles`** (0 except in **I**).

| Scenario | ADC bits | Noise | Disturb cycles | Measured SNR | Theory SNR (ADC) |
|----------|----------|-------|----------------|--------------|------------------|
| A — Ideal | 8 | none | 0 | ~44.2 dB | ~49.9 dB |
| B — Low ADC | 4 | none | 0 | ~21.6 dB | ~25.3 dB |
| C — Thermal | 8 | 0.5% G_max | 0 | ~37.6 dB | — |
| D — Read disturb | 8 | none | 1000 | lower (see CSV) | ~49.9 dB |
| E — Combined | 4 | 0.5% G_max | 1000 | worst in suite (see CSV) | ~25.3 dB |
| F — Multi-layer | 8 | none | 0 | lower than A (L1+L2 ADC; see CSV) | ~49.9 dB |
| G — ReLU | 8 | none | 0 | see CSV | ~49.9 dB |
| H — Sigmoid | 8 | none | 0 | see CSV | ~49.9 dB |
| I — Write endurance | 8 | none | 0 | lower than A (see CSV) | ~49.9 dB |

**Theory column:** classical ADC SQNR: `SQNR ≈ 6.02 × n_bits + 1.76 dB`. The gap vs. Scenario A is expected—the formula assumes a full-scale sine; this demo uses a fixed DC vector.

---

## Defaults (`config.hpp`)

| Constant | Typical value | Role |
|----------|---------------|------|
| `G_max` | 1×10⁻⁴ S | Max cell conductance |
| `G_min` | 1×10⁻⁶ S | Min cell conductance |
| `V_min` | 0.1 V | Min DAC output |
| `V_max` | 1.5 V | Max DAC output |
| `n_bits_adc` | 8 | ADC resolution (overridden per scenario) |
| `disturb_ratio` | 0.03 | Coupling to neighbors |
| `disturb_alpha` | 1×10⁻⁵ | Conductance drift per disturb event |
| `noise_seed` | 42 | Fixed RNG seed for reproducible tests |
| `activation_sigmoid_steepness` | 6 | Sharpness of analog sigmoid (scenario H) |
| `write_endurance_lambda` | 0 (1e−5 in **I**) | Exponent in exp(−λ × cycles); 0 disables scaling |

### JSON overlay (`--config`)

Pass a single JSON **object** whose keys match the table above (same names as in `config.hpp`). Values must be JSON numbers. Unknown keys are ignored. Example:

```json
{ "G_max": 1e-4, "I_min": -6.02e-5, "noise_seed": 42 }
```

### CSV weights (`--weights`), inputs (`--inputs`), second layer (`--weights2`)

**`--weights`:** square **N×N** matrix (1 ≤ N ≤ 512): one row per line, comma-separated numbers in **[-1, 1]** (values outside are clamped, with a warning). Comment lines (`#`) and empty lines follow the same rules as before.

**`--inputs`:** exactly **N** numbers for the DAC path (comma-separated and/or one value per line). Values outside **[0, 1]** are clamped with a warning.

**`--weights2`:** optional second **N×N** matrix for scenario **F** only; must match **N**. If omitted, **F** uses **0.5×I** (helps keep **I_net** in range for the default ADC window).

Without **`--weights`**, the original 4×4 demo matrix and demo input vector are used. With **`--weights`** but without **`--inputs`**, inputs default to a ramp across **[0.15, 0.85]**. For the same numeric behavior as the classic demo while using CSV files, use **`volt.example.weights.csv`** and **`volt.example.inputs.csv`** together.

For arbitrary pretrained weights, tune **`I_min` / `I_range`** (and possibly **`G_max`**) via **`--config`** so the ADC window matches your signal swing.

---

## Project rules

- **`G ≥ 0` always** — violations are fatal.
- **Voltages** stay in `[V_min, V_max]`; out-of-range inputs are not supported.
- **Arithmetic:** `float` in the simulation path; `double` only for reference checks where noted.
- **Tests:** fixed seeds; CI does not accept nondeterministic outputs.
- **Dependencies:** C++17 standard library only.

---

## Roadmap

- [x] Multi-layer chaining (one layer’s ADC → next layer’s DAC) — see scenario `F_multilayer` in `main.cpp`; `SimulatedADC::level_to_dac_normalized` feeds the next DAC.
- [x] Analog activation models (e.g. nonlinear I–V for ReLU / sigmoid) — `activation.hpp`; scenarios `G_relu`, `H_sigmoid`.
- [x] Write endurance (e.g. `G_max` vs. write cycles) — `WriteEnduranceSimulator` in `noise.hpp` / `.cpp`, `CrossbarArray::effective_g_max()`, scenario `I_write_endurance`.
- [x] Benchmark mode (matrix size sweeps, throughput) — `./volt --benchmark`; `benchmark.csv` (GMAC/s, forwards/s).
- [x] JSON config at runtime (no recompile for physics params) — `./volt --config FILE`; `config_json.hpp`; `volt.example.json`.
- [x] CSV weight import for real pretrained weights — `./volt --weights FILE`; optional **`--inputs`**, **`--weights2`**, N×N pipeline; `weights_csv.hpp`; `volt.example.weights.csv` / `volt.example.inputs.csv`.
