#pragma once

#include "config.hpp"

#include <string_view>

namespace volt {

/// Nonlinear crosspoint current: I = G × f(V) depending on [model].
float cell_current(float V, float G, IvModel model, const Config& cfg);

/// Parse JSON / CLI string; returns true if recognized.
bool parse_iv_model(std::string_view name, IvModel& out);

const char* iv_model_name(IvModel model);

}  // namespace volt
