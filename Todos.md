# Project VOLT(Virtual Ohmic-Logic-Testbench): Todos & Implementation Roadmap

## Project Overview
Project VOLT is a software-based physics simulator that emulates the electrical behavior
of an Analog In-Memory Computing (AIMC) crossbar array. The goal is to prove the mathematical
viability of voltage-based neural network inference before committing to physical hardware.
All computation is performed by simulating Ohm's Law (I = V × G) and Kirchhoff's Current Law
(I_total = Σ Vᵢ × Gᵢ) rather than traditional ALU-based MAC operations.

---

## Architectural Decisions (Decided Before Writing Any Code)

### Decision 1: Language — C++17
Native C++ with zero external dependencies. All math is done with the standard library.
Rationale: maximum portability, predictable performance, and no runtime overhead from
third-party packages.

### Decision 2: Negative Weight Encoding — Differential Pair
Physical conductance cannot be negative (G ≥ 0), so each neural network weight is
represented by a pair of cells: G_pos and G_neg.
```
w ∈ [-1, 1]  →  G_pos[i][j] = (w + 1) / 2 * G_max
              →  G_neg[i][j] = (1 - w) / 2 * G_max
I_net = (G_pos - G_neg) * V  =  w * G_max * V
```
This preserves the full signed weight range within the physical constraint of non-negative
conductance.

### Decision 3: ADC Precision — Configurable at Runtime
ADC resolution (n_bits) is a constructor parameter, not a compile-time constant.
The quantization step is computed as:
```
I_step = I_range / (2^n_bits - 1)
level  = clamp(floor(I / I_step), 0, 2^n_bits - 1)
```
This allows direct comparison between 4-bit, 6-bit, and 8-bit configurations in the
same test run.

### Decision 4: ReadDisturb — RRAM Filament Drift Model
Neighboring cell disturbance is modeled as a conductance perturbation proportional to
the voltage applied to the active row:
```
V_dis = V_applied * disturb_ratio   (e.g. 3% of applied voltage)
δG    = alpha * V_dis               (alpha ≈ 1e-5)
```
Only directly adjacent rows (±1) are affected per read operation.

### Decision 5: Validation — Native C++ Reference Implementation
The ground-truth result for every test is computed by a plain double-precision C++ matrix
multiply with no external libraries. Simulation output is compared against this reference
using MSE, max absolute error, and SNR (dB).

---

## Phase 0: Project Infrastructure
- [ ] Create the repository directory layout:
  ```
  VOLT/
  ├── src/
  │   ├── config.hpp
  │   ├── dac.hpp / dac.cpp
  │   ├── adc.hpp / adc.cpp
  │   ├── crossbar.hpp / crossbar.cpp
  │   ├── noise.hpp / noise.cpp
  │   └── main.cpp
  ├── tests/
  │   └── test_equivalence.cpp
  ├── CMakeLists.txt
  └── Todos.md
  ```
- [ ] Write `CMakeLists.txt`: C++17 standard, `-O2` optimization, `-Wall -Wextra` warnings.
- [ ] Write `config.hpp`: a single `Config` struct that holds all physical constants used
      across the project. No magic numbers anywhere else in the codebase.
  ```cpp
  struct Config {
      float G_min       = 1e-6f;   // Siemens
      float G_max       = 1e-4f;   // Siemens
      float V_min       = 0.1f;    // Volts
      float V_max       = 1.5f;    // Volts
      float I_range     = 150e-6f; // Amperes
      int   n_bits_adc  = 8;
      float noise_stddev = 0.0f;
      float disturb_ratio = 0.03f;
      float disturb_alpha = 1e-5f;
      unsigned int noise_seed = 42;
  };
  ```

---

## Phase 1: Core Physical Entities

### 1.1 — SimulatedDAC
Converts digital input values into simulated voltage levels.

- [ ] Create `src/dac.hpp` and `src/dac.cpp`.
- [ ] Constructor takes a `const Config&` reference.
- [ ] Implement `std::vector<float> convert(const std::vector<float>& inputs)`:
  - Inputs are normalized floats in [0.0, 1.0].
  - Formula: `V = V_min + input * (V_max - V_min)`
  - Clamp and log a warning if any input falls outside [0.0, 1.0].
- [ ] Implement overload `convert(const std::vector<uint8_t>& inputs)`:
  - Accepts raw 8-bit values (0–255), normalizes to [0.0, 1.0] first, then converts.
- [ ] Write unit tests:
  - `input = 0.0f  →  output == V_min`
  - `input = 1.0f  →  output == V_max`
  - `input = 0.5f  →  output == (V_min + V_max) / 2`
  - `input = 1.5f  →  output clamped to V_max, warning logged`

### 1.2 — SimulatedADC
Converts accumulated current back into a quantized digital value.

- [ ] Create `src/adc.hpp` and `src/adc.cpp`.
- [ ] Constructor takes a `const Config&` reference.
- [ ] Implement `int quantize(float current)`:
  - Formula: `level = clamp(floor(current / I_step), 0, 2^n_bits - 1)`
  - Where `I_step = I_range / (2^n_bits - 1)`
- [ ] Implement `float reconstruct(int level)`:
  - Inverse of quantize: returns the estimated current for a given digital level.
  - Used to measure quantization error in tests.
- [ ] Write unit tests:
  - 8-bit: `current = I_range / 2  →  level ≈ 127`
  - 4-bit: `current = I_range / 2  →  level ≈ 7`
  - `current = 0.0f               →  level == 0`
  - `current = I_range            →  level == 2^n_bits - 1`
  - `current > I_range            →  level clamped, no crash`

---

## Phase 2: The Crossbar Array

### 2.1 — CrossbarArray Class
The core memory matrix. Stores weights as conductance pairs and performs the physical
matrix-vector multiply.

- [ ] Create `src/crossbar.hpp` and `src/crossbar.cpp`.
- [ ] Constructor: `CrossbarArray(int rows, int cols, const Config& cfg)`.
  - Allocates two 2D matrices internally: `G_pos[rows][cols]` and `G_neg[rows][cols]`.
  - All conductance values initialized to `G_min`.
- [ ] Implement `void load_weights(const std::vector<std::vector<float>>& weights)`:
  - Expects weights in range [-1.0, 1.0].
  - Applies the differential pair mapping for each element:
    ```
    G_pos[i][j] = ((w + 1.0f) / 2.0f) * G_max
    G_neg[i][j] = ((1.0f - w) / 2.0f) * G_max
    ```
  - If a weight is outside [-1.0, 1.0], clamp it and log a warning.
  - Throw `std::invalid_argument` if matrix dimensions don't match constructor parameters.
- [ ] Implement `float get_effective_weight(int i, int j) const`:
  - Returns `(G_pos[i][j] - G_neg[i][j]) / G_max`.
  - Used to verify that the loaded weights can be accurately recovered.
- [ ] Write unit tests:
  - `w =  1.0f  →  G_pos = G_max, G_neg = 0`
  - `w = -1.0f  →  G_pos = 0,     G_neg = G_max`
  - `w =  0.0f  →  G_pos = G_neg = G_max / 2`
  - `get_effective_weight` recovers the original weight within float epsilon.

### 2.2 — apply_voltage() — The Core Engine
Simulates the physical matrix-vector multiply using Ohm's Law and KCL.

- [ ] Implement `std::vector<float> apply_voltage(const std::vector<float>& voltages)`:
  1. Validate that `voltages.size() == rows`. Throw `std::invalid_argument` otherwise.
  2. Initialize `I_pos[cols]` and `I_neg[cols]` to zero.
  3. Outer loop over columns, inner loop over rows (cache-friendly access pattern):
     ```cpp
     for (int j = 0; j < cols; j++)
         for (int i = 0; i < rows; i++) {
             I_pos[j] += voltages[i] * G_pos[i][j];
             I_neg[j] += voltages[i] * G_neg[i][j];
         }
     ```
  4. Compute `I_net[j] = I_pos[j] - I_neg[j]` for each column.
  5. Return `I_net`.
- [ ] Write unit tests:
  - A 2×2 identity weight matrix with a known voltage vector must produce the exact voltage
    values as output currents (scaled by G_max).

---

## Phase 3: Noise & Real-World Physics

### 3.1 — ThermalNoiseInjector
Models heat-induced random fluctuations in cell conductance.

- [ ] Create `src/noise.hpp` and `src/noise.cpp`.
- [ ] Constructor: `ThermalNoiseInjector(const Config& cfg)`.
  - Seeds a `std::mt19937` RNG with `cfg.noise_seed`.
  - Uses a `std::normal_distribution<float>(0.0f, cfg.noise_stddev)`.
- [ ] Implement two injection modes:
  - `void inject_transient(std::vector<float>& conductances)`:
    Adds noise to a temporary copy used only for the current read. Original G values
    are not modified.
  - `void inject_persistent(CrossbarArray& array)`:
    Adds noise directly to the G_pos and G_neg matrices (simulates permanent drift).
    Clamps all values to [0, G_max] after injection.
- [ ] Write unit tests:
  - With `stddev = 0.0f`, output must be identical to input.
  - With fixed seed, two consecutive calls must produce identical noise sequences.
  - After injection, no conductance value may be negative.

### 3.2 — ReadDisturbSimulator
Models the RRAM filament drift caused by repeated read operations.

- [ ] Constructor: `ReadDisturbSimulator(const Config& cfg)`.
- [ ] Implement `void apply_disturb(CrossbarArray& array, int active_row, float V_applied)`:
  - Compute `V_dis = V_applied * disturb_ratio`.
  - For each column j, apply to neighbor rows (active_row ± 1, if they exist):
    ```
    G_pos[neighbor][j] += alpha * V_dis
    G_neg[neighbor][j] += alpha * V_dis
    ```
  - Clamp all modified values to [0, G_max].
- [ ] Implement `void log_drift_report(const CrossbarArray& array)`:
  - Computes the average and max conductance shift across all cells compared to the
    originally loaded weights. Prints a summary to stdout.
- [ ] Write unit tests:
  - After 1000 calls to `apply_disturb` on the same row, verify that the neighbor
    cell conductance shift is within the analytically expected range:
    `delta_G ≈ 1000 * alpha * V_applied * disturb_ratio`

---

## Phase 4: Integration & Validation

### 4.1 — Full Pipeline Test (main.cpp)
- [ ] Instantiate a `Config` with default values.
- [ ] Define a fixed 4×4 weight matrix:
  ```
  W = [[ 0.8, -0.3,  0.5, -0.1],
       [-0.6,  0.9, -0.2,  0.7],
       [ 0.1, -0.8,  0.4, -0.5],
       [ 0.3,  0.2, -0.9,  0.6]]
  ```
- [ ] Define a fixed input vector: `[0.9, 0.4, 0.7, 0.2]`
- [ ] Run the complete pipeline for each scenario:
  ```
  input vector
      → SimulatedDAC    (digital → voltage)
      → CrossbarArray   (voltage × conductance → current)
      → [optional noise injectors]
      → SimulatedADC    (current → digital)
      → output vector
  ```
- [ ] Compute the reference result using a plain C++ double-precision matrix multiply.
- [ ] For each scenario, compute and print:
  - **MSE** — Mean Squared Error between simulation output and reference.
  - **Max Absolute Error** — largest single-element deviation.
  - **SNR (dB)** — `10 * log10(signal_power / noise_power)`

### 4.2 — Comparison Scenarios
- [ ] **Scenario A — Ideal:**
  No noise, 8-bit ADC. Expected MSE < 1e-6. This is the baseline.
- [ ] **Scenario B — Low ADC Precision:**
  No noise, 4-bit ADC. Measure how much quantization alone degrades SNR.
- [ ] **Scenario C — Thermal Noise:**
  8-bit ADC, `noise_stddev = 0.5% of G_max`. Is accuracy still acceptable?
- [ ] **Scenario D — Read Disturb:**
  8-bit ADC, no thermal noise. Run `apply_disturb` for 1000 cycles before reading.
  Measure accumulated drift impact.
- [ ] **Scenario E — Combined:**
  4-bit ADC + thermal noise + read disturb all active simultaneously. Worst-case
  estimate of real-world analog accuracy.
- [ ] Write all scenario results to `results.csv`:
  ```
  scenario, n_bits, noise_stddev, disturb_cycles, mse, max_abs_err, snr_db
  ```

### 4.3 — Automated Regression Tests (test_equivalence.cpp)
- [ ] Assert that Scenario A (ideal) produces MSE < 1e-6. Fail the build if exceeded.
- [ ] Assert that all conductance values remain in [0, G_max] after every operation.
- [ ] Assert that ADC output never exceeds `2^n_bits - 1`.
- [ ] Integrate with CTest: `cmake --build . && ctest --output-on-failure`.

---

## Phase 5: Optional Future Extensions
- [ ] **Multi-layer chaining:** Feed the ADC output of one `CrossbarArray` as the DAC
      input of the next. Simulate a two-layer MLP end-to-end.
- [ ] **Analog activation functions:** Model ReLU or sigmoid as a nonlinear I-V curve
      applied between layers.
- [ ] **Write endurance degradation:** Track how many write cycles each cell has undergone
      and progressively degrade its G_max accordingly.
- [ ] **Benchmark mode:** Sweep matrix sizes from 4×4 to 512×512 and log throughput,
      measuring how simulation time scales with array size.
- [ ] **JSON config loader:** Load `Config` from a `config.json` file at startup so
      physical parameters can be changed without recompiling.
- [ ] **CSV weight loader:** Accept weight matrices from `.csv` files to test with real
      pre-trained model weights instead of hand-written test values.

---

## Hard Rules (Non-Negotiable)
> All conductance values must satisfy G ≥ 0 at all times. Any violation is a fatal error.
> All voltage values must be clamped to [V_min, V_max] before being applied to the array.
> All simulation arithmetic uses float32. double is reserved for reference calculations only.
> All noise seeds are fixed constants in test files. Non-deterministic tests are not allowed.
> No external libraries. Only the C++17 standard library is permitted.
