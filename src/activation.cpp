#include "activation.hpp"

#include <algorithm>
#include <cmath>

namespace volt {

float apply_activation(float I, Activation a, const Config& cfg) {
    switch (a) {
        case Activation::Identity:
            return I;
        case Activation::ReLU:
            return std::max(0.0f, I);
        case Activation::Sigmoid: {
            const float mid = cfg.I_min + 0.5f * cfg.I_range;
            const float scale = cfg.I_range * 0.25f + 1e-30f;
            const float x = (I - mid) / scale * cfg.activation_sigmoid_steepness;
            const float s = 1.0f / (1.0f + std::exp(-x));
            return cfg.I_min + cfg.I_range * s;
        }
        default:
            return I;
    }
}

double apply_activation(double I, Activation a, const Config& cfg) {
    switch (a) {
        case Activation::Identity:
            return I;
        case Activation::ReLU:
            return std::max(0.0, I);
        case Activation::Sigmoid: {
            const double mid = static_cast<double>(cfg.I_min) + 0.5 * static_cast<double>(cfg.I_range);
            const double scale = static_cast<double>(cfg.I_range) * 0.25 + 1e-30;
            const double x =
                (I - mid) / scale * static_cast<double>(cfg.activation_sigmoid_steepness);
            const double s = 1.0 / (1.0 + std::exp(-x));
            return static_cast<double>(cfg.I_min) + static_cast<double>(cfg.I_range) * s;
        }
        default:
            return I;
    }
}

}  // namespace volt
