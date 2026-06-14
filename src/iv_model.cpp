#include "iv_model.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>
#include <string_view>

namespace volt {

namespace {

float sign_v(float v) {
    if (v > 0.0f) {
        return 1.0f;
    }
    if (v < 0.0f) {
        return -1.0f;
    }
    return 0.0f;
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

}  // namespace

float cell_current(float V, float G, IvModel model, const Config& cfg) {
    switch (model) {
        case IvModel::Linear:
            return G * V;
        case IvModel::PowerLaw: {
            const float v_ref = std::max(cfg.iv_v_ref, 1e-12f);
            const float alpha = cfg.iv_exponent;
            const float mag = std::pow(std::abs(V) / v_ref, alpha);
            return G * sign_v(V) * mag * v_ref;
        }
        case IvModel::SoftSaturation: {
            const float v_sat = std::max(cfg.iv_v_sat, 1e-12f);
            return G * V / (1.0f + std::abs(V) / v_sat);
        }
        default:
            return G * V;
    }
}

bool parse_iv_model(std::string_view name, IvModel& out) {
    const std::string k = lower(name);
    if (k == "linear") {
        out = IvModel::Linear;
        return true;
    }
    if (k == "power_law" || k == "powerlaw") {
        out = IvModel::PowerLaw;
        return true;
    }
    if (k == "soft_saturation" || k == "softsaturation") {
        out = IvModel::SoftSaturation;
        return true;
    }
    return false;
}

const char* iv_model_name(IvModel model) {
    switch (model) {
        case IvModel::Linear:
            return "linear";
        case IvModel::PowerLaw:
            return "power_law";
        case IvModel::SoftSaturation:
            return "soft_saturation";
        default:
            return "linear";
    }
}

}  // namespace volt
