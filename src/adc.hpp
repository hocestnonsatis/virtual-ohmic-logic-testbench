#pragma once

#include "config.hpp"

namespace volt {

class SimulatedADC {
public:
    explicit SimulatedADC(const Config& cfg);

    int quantize(float current);
    float reconstruct(int level) const;

    int max_level() const;
    float i_step() const;

private:
    Config cfg_;
};

}  // namespace volt
