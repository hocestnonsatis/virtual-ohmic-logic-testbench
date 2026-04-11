# VOLT — Virtual Ohmic Logic Testbench

Software physics simulator for **Analog In-Memory Computing (AIMC)** crossbar arrays. VOLT models voltage-based neural inference with Ohm’s law and Kirchhoff’s current law—useful for exploring feasibility before hardware exists.

**Stack:** C++17, CMake ≥ 3.14, standard library only (no third-party dependencies).

---

## Contents

1. [Concept](#concept) — digital operations vs. simulated analog  
2. [Quick start](#quick-start) — build, run, tests  
3. [Repository layout](#repository-layout)  
4. [Design notes](#design-notes) — differential pair, bipolar ADC, read disturb  
5. [Scenarios & results](#scenarios--results)  
6. [Defaults (`config.hpp`)](#defaults-confighpp)  
7. [Project rules](#project-rules)  
8. [Roadmap](#roadmap)

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

Run tests:

```bash
cd build && ctest --output-on-failure
```

---

## Repository layout

```
.
├── src/
│   ├── config.hpp          # Physical constants
│   ├── dac.hpp / dac.cpp   # Digital → voltage
│   ├── adc.hpp / adc.cpp   # Current → digital
│   ├── crossbar.hpp / .cpp # Weight matrix (differential pair)
│   ├── noise.hpp / .cpp    # Thermal noise + read disturb
│   └── main.cpp            # Pipeline + scenarios A–E
├── tests/
│   ├── test_core.cpp
│   └── test_equivalence.cpp  # Regression: MSE < 1e-6 (ideal path)
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

---

## Scenarios & results

Five scenarios use a fixed 4×4 weight matrix and input vector. Output is **`results.csv`** in the working directory when you run `./volt` (typically `build/results.csv`).

| Scenario | ADC bits | Noise | Disturb cycles | Measured SNR | Theory SNR (ADC) |
|----------|----------|-------|----------------|--------------|------------------|
| A — Ideal | 8 | none | 0 | ~44.2 dB | ~49.9 dB |
| B — Low ADC | 4 | none | 0 | ~21.6 dB | ~25.3 dB |
| C — Thermal | 8 | 0.5% G_max | 0 | ~37.6 dB | — |
| D — Read disturb | 8 | none | 1000 | lower (see CSV) | ~49.9 dB |
| E — Combined | 4 | 0.5% G_max | 1000 | worst in suite (see CSV) | ~25.3 dB |

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

---

## Project rules

- **`G ≥ 0` always** — violations are fatal.
- **Voltages** stay in `[V_min, V_max]`; out-of-range inputs are not supported.
- **Arithmetic:** `float` in the simulation path; `double` only for reference checks where noted.
- **Tests:** fixed seeds; CI does not accept nondeterministic outputs.
- **Dependencies:** C++17 standard library only.

---

## Roadmap

- [ ] Multi-layer chaining (one layer’s ADC → next layer’s DAC)
- [ ] Analog activation models (e.g. nonlinear I–V for ReLU / sigmoid)
- [ ] Write endurance (e.g. `G_max` vs. write cycles)
- [ ] Benchmark mode (matrix size sweeps, throughput)
- [ ] JSON config at runtime (no recompile for physics params)
- [ ] CSV weight import for real pretrained weights
