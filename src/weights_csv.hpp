#pragma once

#include <string>
#include <vector>

namespace volt {

/// Reads a weight matrix (N×M): one row per line, comma-separated numbers.
/// Lines starting with `#` (after spaces) are comments; empty lines are skipped.
/// Values are clamped to [-1, 1] with the same semantics as `CrossbarArray::load_weights`.
/// Dimensions must satisfy 1 <= N <= k_max_weights_rows and 1 <= M <= k_max_weights_cols.
bool load_weights_csv_file(const std::string& path, std::vector<std::vector<double>>& out,
                           std::string& err);

/// Maximum supported row count N for an N×M weight matrix.
constexpr int k_max_weights_rows = 512;
/// Maximum supported column count M for an N×M weight matrix.
constexpr int k_max_weights_cols = 512;

/// Reads a normalized input vector for the DAC: comma-separated and/or one value per line.
/// Total count must equal [expected_n]. Values are clamped to [0, 1] with optional warnings.
bool load_inputs_csv_file(const std::string& path, int expected_n, std::vector<float>& out,
                          std::string& err);

}  // namespace volt
