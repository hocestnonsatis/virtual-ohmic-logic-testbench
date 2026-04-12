#pragma once

#include <string>
#include <vector>

namespace volt {

/// Reads a square weight matrix: one row per line, comma-separated numbers.
/// Lines starting with `#` (after spaces) are comments; empty lines are skipped.
/// Values are clamped to [-1, 1] with the same semantics as `CrossbarArray::load_weights`.
/// Dimension must be square and in [1, k_max_weights_dim].
bool load_weights_csv_file(const std::string& path, std::vector<std::vector<double>>& out,
                           std::string& err);

/// Maximum supported N for an N×N weight matrix (guardrail for CLI / file mistakes).
constexpr int k_max_weights_dim = 512;

/// Reads a normalized input vector for the DAC: comma-separated and/or one value per line.
/// Total count must equal [expected_n]. Values are clamped to [0, 1] with optional warnings.
bool load_inputs_csv_file(const std::string& path, int expected_n, std::vector<float>& out,
                          std::string& err);

}  // namespace volt
