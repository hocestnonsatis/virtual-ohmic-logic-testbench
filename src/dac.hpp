#pragma once

#include "config.hpp"

#include <cstdint>
#include <vector>

namespace volt {

class SimulatedDAC {
public:
    explicit SimulatedDAC(const Config& cfg);

    std::vector<float> convert(const std::vector<float>& inputs);
    std::vector<float> convert(const std::vector<std::uint8_t>& inputs);

    float v_min() const { return cfg_.V_min; }
    float v_max() const { return cfg_.V_max; }

private:
    Config cfg_;
};

}  // namespace volt
