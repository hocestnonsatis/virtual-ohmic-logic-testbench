#pragma once

#include <string>
#include <vector>

namespace volt {

/// Reads a square weight matrix: one row per line, comma-separated numbers.
/// Lines starting with `#` (after spaces) are comments; empty lines are skipped.
/// Values are clamped to [-1, 1] with the same semantics as `CrossbarArray::load_weights`.
/// Currently only 4×4 matrices are accepted (matches the demo pipeline).
bool load_weights_csv_file(const std::string& path, std::vector<std::vector<double>>& out,
                           std::string& err);

}  // namespace volt
