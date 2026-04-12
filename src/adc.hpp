#pragma once

#include "config.hpp"

namespace volt {

class SimulatedADC {
public:
    explicit SimulatedADC(const Config& cfg);

    int quantize(float current);
    float reconstruct(int level) const;

    /// Maps an ADC code to [0, 1] for the next layer's DAC (multi-layer: ADC → DAC).
    float level_to_dac_normalized(int level) const;

    int max_level() const;
    float i_step() const;

private:
    Config cfg_;
};

}  // namespace volt
