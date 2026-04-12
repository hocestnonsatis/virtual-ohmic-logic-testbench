#include "adc.hpp"

#include <algorithm>
#include <cmath>

namespace volt {

SimulatedADC::SimulatedADC(const Config& cfg) : cfg_(cfg) {}

int SimulatedADC::max_level() const {
    int n = cfg_.n_bits_adc;
    if (n < 1) {
        return 0;
    }
    return (1 << n) - 1;
}

float SimulatedADC::i_step() const {
    int denom = max_level();
    if (denom <= 0) {
        return cfg_.I_range;
    }
    return cfg_.I_range / static_cast<float>(denom);
}

int SimulatedADC::quantize(float current) {
    float step = i_step();
    if (step <= 0.0f) {
        return 0;
    }
    float shifted = current - cfg_.I_min;
    float raw = std::floor(shifted / step);
    int level = static_cast<int>(raw);
    level = std::clamp(level, 0, max_level());
    return level;
}

float SimulatedADC::reconstruct(int level) const {
    level = std::clamp(level, 0, max_level());
    return cfg_.I_min + static_cast<float>(level) * i_step();
}

float SimulatedADC::level_to_dac_normalized(int level) const {
    const int mx = max_level();
    if (mx <= 0) {
        return 0.0f;
    }
    level = std::clamp(level, 0, mx);
    return static_cast<float>(level) / static_cast<float>(mx);
}

}  // namespace volt
