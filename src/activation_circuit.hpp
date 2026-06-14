#pragma once

#include "config.hpp"

#include <string_view>

namespace volt {

/// Physical inter-layer current transfer (I_in → I_out).
float circuit_transfer(float I_in, CircuitModel model, const Config& cfg);
double circuit_transfer(double I_in, CircuitModel model, const Config& cfg);

bool parse_circuit_model(std::string_view name, CircuitModel& out);
const char* circuit_model_name(CircuitModel model);

}  // namespace volt
