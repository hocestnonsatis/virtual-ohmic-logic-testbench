#pragma once

#include "config.hpp"

namespace volt {

/// Time crossbar MAC forwards for increasing n×n sizes; writes `benchmark.csv` in cwd.
void run_benchmark_suite(const Config& cfg);

}  // namespace volt
