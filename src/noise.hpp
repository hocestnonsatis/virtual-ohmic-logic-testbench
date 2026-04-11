#pragma once

#include "config.hpp"
#include "crossbar.hpp"

#include <random>
#include <vector>

namespace volt {

class ThermalNoiseInjector {
public:
    explicit ThermalNoiseInjector(const Config& cfg);

    void inject_transient(std::vector<float>& conductances);
    void inject_persistent(CrossbarArray& array);

private:
    Config cfg_;
    std::mt19937 rng_;
    std::normal_distribution<float> dist_;
};

class ReadDisturbSimulator {
public:
    explicit ReadDisturbSimulator(const Config& cfg);

    void apply_disturb(CrossbarArray& array, int active_row, float V_applied);
    void log_drift_report(const CrossbarArray& array) const;

private:
    Config cfg_;
};

}  // namespace volt
