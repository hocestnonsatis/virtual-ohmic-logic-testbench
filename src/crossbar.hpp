#pragma once

#include "config.hpp"

#include <stdexcept>
#include <vector>

namespace volt {

class ThermalNoiseInjector;
class ReadDisturbSimulator;
class WriteEnduranceSimulator;

class CrossbarArray {
public:
    CrossbarArray(int rows, int cols, const Config& cfg);

    void load_weights(const std::vector<std::vector<float>>& weights);

    float get_effective_weight(int i, int j) const;

    std::vector<float> apply_voltage(const std::vector<float>& voltages);

    /// Same physics as apply_voltage but uses supplied conductance matrices (e.g. after
    /// transient noise on a copy). Dimensions must match rows/cols.
    std::vector<float> apply_voltage(const std::vector<float>& voltages,
                                     const std::vector<std::vector<float>>& G_pos,
                                     const std::vector<std::vector<float>>& G_neg) const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const Config& config() const { return cfg_; }

    /// For tests / drift: raw conductance at cell (read-only).
    float g_pos_at(int i, int j) const;
    float g_neg_at(int i, int j) const;

    /// Effective ceiling after write-endurance scaling (nominal `cfg.G_max` before stress).
    float effective_g_max() const { return g_max_effective_; }

private:
    friend class ThermalNoiseInjector;
    friend class ReadDisturbSimulator;
    friend class WriteEnduranceSimulator;

    void apply_uniform_conductance_scale(float scale);

    int rows_;
    int cols_;
    Config cfg_;
    float g_max_effective_;
    std::vector<std::vector<float>> G_pos_;
    std::vector<std::vector<float>> G_neg_;
    /// Snapshot after last successful load_weights (for drift reporting).
    std::vector<std::vector<float>> G_pos_baseline_;
    std::vector<std::vector<float>> G_neg_baseline_;
};

}  // namespace volt
