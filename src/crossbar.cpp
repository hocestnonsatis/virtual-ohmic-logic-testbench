#include "crossbar.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace volt {

CrossbarArray::CrossbarArray(int rows, int cols, const Config& cfg)
    : rows_(rows), cols_(cols), cfg_(cfg), g_max_effective_(cfg_.G_max) {
    G_pos_.assign(static_cast<std::size_t>(rows),
                  std::vector<float>(static_cast<std::size_t>(cols), cfg_.G_min));
    G_neg_.assign(static_cast<std::size_t>(rows),
                  std::vector<float>(static_cast<std::size_t>(cols), cfg_.G_min));
    G_pos_baseline_ = G_pos_;
    G_neg_baseline_ = G_neg_;
}

void CrossbarArray::load_weights(const std::vector<std::vector<float>>& weights) {
    if (static_cast<int>(weights.size()) != rows_) {
        throw std::invalid_argument("CrossbarArray::load_weights: row count mismatch");
    }
    for (int i = 0; i < rows_; ++i) {
        if (static_cast<int>(weights[static_cast<std::size_t>(i)].size()) != cols_) {
            throw std::invalid_argument("CrossbarArray::load_weights: column count mismatch");
        }
    }
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            float w = weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            if (w < -1.0f || w > 1.0f) {
                std::cerr << "[CrossbarArray] warning: weight " << w
                          << " outside [-1,1]; clamping\n";
                w = std::clamp(w, -1.0f, 1.0f);
            }
            G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                ((w + 1.0f) / 2.0f) * cfg_.G_max;
            G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                ((1.0f - w) / 2.0f) * cfg_.G_max;
        }
    }
    g_max_effective_ = cfg_.G_max;
    G_pos_baseline_ = G_pos_;
    G_neg_baseline_ = G_neg_;
}

float CrossbarArray::get_effective_weight(int i, int j) const {
    float gp = G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    float gn = G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return (gp - gn) / cfg_.G_max;
}

std::vector<float> CrossbarArray::apply_voltage(const std::vector<float>& voltages) {
    if (static_cast<int>(voltages.size()) != rows_) {
        throw std::invalid_argument("CrossbarArray::apply_voltage: voltage count != rows");
    }
    std::vector<float> v_row(static_cast<std::size_t>(rows_));
    for (int i = 0; i < rows_; ++i) {
        v_row[static_cast<std::size_t>(i)] =
            std::clamp(voltages[static_cast<std::size_t>(i)], cfg_.V_min, cfg_.V_max);
    }
    std::vector<float> I_pos(static_cast<std::size_t>(cols_), 0.0f);
    std::vector<float> I_neg(static_cast<std::size_t>(cols_), 0.0f);
    for (int j = 0; j < cols_; ++j) {
        for (int i = 0; i < rows_; ++i) {
            float v = v_row[static_cast<std::size_t>(i)];
            I_pos[static_cast<std::size_t>(j)] +=
                v * G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            I_neg[static_cast<std::size_t>(j)] +=
                v * G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    std::vector<float> I_net(static_cast<std::size_t>(cols_));
    for (int j = 0; j < cols_; ++j) {
        I_net[static_cast<std::size_t>(j)] =
            I_pos[static_cast<std::size_t>(j)] - I_neg[static_cast<std::size_t>(j)];
    }
    return I_net;
}

std::vector<float> CrossbarArray::apply_voltage(
    const std::vector<float>& voltages,
    const std::vector<std::vector<float>>& G_pos,
    const std::vector<std::vector<float>>& G_neg) const {
    if (static_cast<int>(voltages.size()) != rows_) {
        throw std::invalid_argument("CrossbarArray::apply_voltage: voltage count != rows");
    }
    if (static_cast<int>(G_pos.size()) != rows_ || static_cast<int>(G_neg.size()) != rows_) {
        throw std::invalid_argument("CrossbarArray::apply_voltage: conductance row mismatch");
    }
    for (int i = 0; i < rows_; ++i) {
        if (static_cast<int>(G_pos[static_cast<std::size_t>(i)].size()) != cols_ ||
            static_cast<int>(G_neg[static_cast<std::size_t>(i)].size()) != cols_) {
            throw std::invalid_argument("CrossbarArray::apply_voltage: conductance col mismatch");
        }
    }
    std::vector<float> v_row(static_cast<std::size_t>(rows_));
    for (int i = 0; i < rows_; ++i) {
        v_row[static_cast<std::size_t>(i)] =
            std::clamp(voltages[static_cast<std::size_t>(i)], cfg_.V_min, cfg_.V_max);
    }
    std::vector<float> I_pos(static_cast<std::size_t>(cols_), 0.0f);
    std::vector<float> I_neg(static_cast<std::size_t>(cols_), 0.0f);
    for (int j = 0; j < cols_; ++j) {
        for (int i = 0; i < rows_; ++i) {
            float v = v_row[static_cast<std::size_t>(i)];
            I_pos[static_cast<std::size_t>(j)] +=
                v * G_pos[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            I_neg[static_cast<std::size_t>(j)] +=
                v * G_neg[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    std::vector<float> I_net(static_cast<std::size_t>(cols_));
    for (int j = 0; j < cols_; ++j) {
        I_net[static_cast<std::size_t>(j)] =
            I_pos[static_cast<std::size_t>(j)] - I_neg[static_cast<std::size_t>(j)];
    }
    return I_net;
}

float CrossbarArray::g_pos_at(int i, int j) const {
    return G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}

float CrossbarArray::g_neg_at(int i, int j) const {
    return G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}

void CrossbarArray::apply_uniform_conductance_scale(float scale) {
    scale = std::clamp(scale, 0.0f, 1.0f);
    g_max_effective_ = cfg_.G_max * scale;
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            float& gp = G_pos_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            float& gn = G_neg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            gp *= scale;
            gn *= scale;
            gp = std::clamp(gp, 0.0f, g_max_effective_);
            gn = std::clamp(gn, 0.0f, g_max_effective_);
        }
    }
    G_pos_baseline_ = G_pos_;
    G_neg_baseline_ = G_neg_;
}

}  // namespace volt
