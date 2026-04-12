#include "noise.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace volt {

ThermalNoiseInjector::ThermalNoiseInjector(const Config& cfg)
    : cfg_(cfg), rng_(cfg.noise_seed), dist_(0.0f, cfg.noise_stddev) {}

void ThermalNoiseInjector::inject_transient(std::vector<float>& conductances) {
    if (cfg_.noise_stddev <= 0.0f) {
        return;
    }
    for (float& g : conductances) {
        g += dist_(rng_);
    }
}

void ThermalNoiseInjector::inject_persistent(CrossbarArray& array) {
    if (cfg_.noise_stddev <= 0.0f) {
        return;
    }
    const float gmax = array.effective_g_max();
    for (int i = 0; i < array.rows(); ++i) {
        for (int j = 0; j < array.cols(); ++j) {
            float np = dist_(rng_);
            float nn = dist_(rng_);
            array.G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                std::clamp(array.G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] + np,
                           0.0f, gmax);
            array.G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                std::clamp(array.G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] + nn,
                           0.0f, gmax);
        }
    }
}

ReadDisturbSimulator::ReadDisturbSimulator(const Config& cfg) : cfg_(cfg) {}

void ReadDisturbSimulator::apply_disturb(CrossbarArray& array, int active_row, float V_applied) {
    if (active_row < 0 || active_row >= array.rows()) {
        return;
    }
    float V_dis = V_applied * cfg_.disturb_ratio;
    float delta = cfg_.disturb_alpha * V_dis;
    const float gmax = array.effective_g_max();
    for (int neighbor : {active_row - 1, active_row + 1}) {
        if (neighbor < 0 || neighbor >= array.rows()) {
            continue;
        }
        for (int j = 0; j < array.cols(); ++j) {
            array.G_pos_[static_cast<std::size_t>(neighbor)][static_cast<std::size_t>(j)] =
                std::clamp(
                    array.G_pos_[static_cast<std::size_t>(neighbor)][static_cast<std::size_t>(j)] +
                        delta,
                    0.0f, gmax);
            array.G_neg_[static_cast<std::size_t>(neighbor)][static_cast<std::size_t>(j)] =
                std::clamp(
                    array.G_neg_[static_cast<std::size_t>(neighbor)][static_cast<std::size_t>(j)] +
                        delta,
                    0.0f, gmax);
        }
    }
}

void ReadDisturbSimulator::log_drift_report(const CrossbarArray& array) const {
    double sum_shift = 0.0;
    double max_shift = 0.0;
    int n = array.rows() * array.cols();
    if (n == 0) {
        std::cout << "[ReadDisturb] drift report: empty array\n";
        return;
    }
    for (int i = 0; i < array.rows(); ++i) {
        for (int j = 0; j < array.cols(); ++j) {
            float gp0 = array.G_pos_baseline_[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(j)];
            float gn0 = array.G_neg_baseline_[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(j)];
            float gp = array.G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            float gn = array.G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            double shift_pos = std::abs(static_cast<double>(gp) - static_cast<double>(gp0));
            double shift_neg = std::abs(static_cast<double>(gn) - static_cast<double>(gn0));
            double cell_shift = std::max(shift_pos, shift_neg);
            sum_shift += cell_shift;
            max_shift = std::max(max_shift, cell_shift);
        }
    }
    double avg = sum_shift / static_cast<double>(n);
    std::cout << "[ReadDisturb] drift: avg conductance shift = " << avg
              << " S, max = " << max_shift << " S\n";
}

WriteEnduranceSimulator::WriteEnduranceSimulator(const Config& cfg) : cfg_(cfg) {}

void WriteEnduranceSimulator::apply_write_cycles(CrossbarArray& array, int cycles) {
    if (cycles <= 0 || cfg_.write_endurance_lambda <= 0.0f) {
        return;
    }
    const float scale =
        std::exp(-cfg_.write_endurance_lambda * static_cast<float>(cycles));
    array.apply_uniform_conductance_scale(scale);
}

}  // namespace volt
