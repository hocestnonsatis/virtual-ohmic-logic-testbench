#include "dac.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace volt {

SimulatedDAC::SimulatedDAC(const Config& cfg) : cfg_(cfg) {}

std::vector<float> SimulatedDAC::convert(const std::vector<float>& inputs) {
    std::vector<float> out;
    out.reserve(inputs.size());
    for (float x : inputs) {
        float clamped = x;
        if (x < 0.0f || x > 1.0f) {
            std::cerr << "[SimulatedDAC] warning: input " << x
                      << " outside [0,1]; clamping to [0,1]\n";
            clamped = std::clamp(x, 0.0f, 1.0f);
        }
        out.push_back(cfg_.V_min + clamped * (cfg_.V_max - cfg_.V_min));
    }
    return out;
}

std::vector<float> SimulatedDAC::convert(const std::vector<std::uint8_t>& inputs) {
    std::vector<float> normalized;
    normalized.reserve(inputs.size());
    for (std::uint8_t b : inputs) {
        normalized.push_back(static_cast<float>(b) / 255.0f);
    }
    return convert(normalized);
}

}  // namespace volt
