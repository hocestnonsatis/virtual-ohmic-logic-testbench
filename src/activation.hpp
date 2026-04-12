#pragma once

#include "config.hpp"

namespace volt {

enum class Activation { Identity, ReLU, Sigmoid };

float apply_activation(float I, Activation a, const Config& cfg);
double apply_activation(double I, Activation a, const Config& cfg);

}  // namespace volt
