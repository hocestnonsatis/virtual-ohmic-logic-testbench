#include "activation_circuit.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>
#include <string_view>

namespace volt {

namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

float sigmoid_circuit(float I_in, const Config& cfg) {
    const float mid = cfg.I_min + 0.5f * cfg.I_range;
    const float scale = cfg.I_range * 0.25f + 1e-30f;
    const float x = (I_in - mid) / scale * cfg.circuit_steepness;
    const float s = 1.0f / (1.0f + std::exp(-x));
    return cfg.I_min + cfg.I_range * s;
}

double sigmoid_circuit(double I_in, const Config& cfg) {
    const double mid = static_cast<double>(cfg.I_min) + 0.5 * static_cast<double>(cfg.I_range);
    const double scale = static_cast<double>(cfg.I_range) * 0.25 + 1e-30;
    const double x =
        (I_in - mid) / scale * static_cast<double>(cfg.circuit_steepness);
    const double s = 1.0 / (1.0 + std::exp(-x));
    return static_cast<double>(cfg.I_min) + static_cast<double>(cfg.I_range) * s;
}

}  // namespace

float circuit_transfer(float I_in, CircuitModel model, const Config& cfg) {
    switch (model) {
        case CircuitModel::PassThrough:
            return I_in;
        case CircuitModel::DiodeRectifier:
            return std::max(0.0f, I_in - cfg.circuit_i_threshold);
        case CircuitModel::TunableSigmoid:
            return sigmoid_circuit(I_in, cfg);
        default:
            return I_in;
    }
}

double circuit_transfer(double I_in, CircuitModel model, const Config& cfg) {
    switch (model) {
        case CircuitModel::PassThrough:
            return I_in;
        case CircuitModel::DiodeRectifier:
            return std::max(0.0, I_in - static_cast<double>(cfg.circuit_i_threshold));
        case CircuitModel::TunableSigmoid:
            return sigmoid_circuit(I_in, cfg);
        default:
            return I_in;
    }
}

bool parse_circuit_model(std::string_view name, CircuitModel& out) {
    const std::string k = lower(name);
    if (k == "pass_through" || k == "passthrough" || k == "identity") {
        out = CircuitModel::PassThrough;
        return true;
    }
    if (k == "diode_rectifier" || k == "diode" || k == "relu") {
        out = CircuitModel::DiodeRectifier;
        return true;
    }
    if (k == "tunable_sigmoid" || k == "sigmoid") {
        out = CircuitModel::TunableSigmoid;
        return true;
    }
    return false;
}

const char* circuit_model_name(CircuitModel model) {
    switch (model) {
        case CircuitModel::PassThrough:
            return "pass_through";
        case CircuitModel::DiodeRectifier:
            return "diode_rectifier";
        case CircuitModel::TunableSigmoid:
            return "tunable_sigmoid";
        default:
            return "pass_through";
    }
}

}  // namespace volt
