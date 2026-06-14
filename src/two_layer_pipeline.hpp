#pragma once

#include "activation.hpp"
#include "config.hpp"

#include <string>
#include <vector>

namespace volt {

struct TwoLayerOptions {
    Activation interlayer_activation = Activation::Identity;
    CircuitModel interlayer_circuit = CircuitModel::PassThrough;
    bool use_transient_noise = false;
    bool use_persistent_noise = false;
    int disturb_cycles = 0;
    int write_endurance_cycles = 0;
};

struct TwoLayerResult {
    std::string name;
    int n_bits = 0;
    float noise_stddev = 0.0f;
    int disturb_cycles = 0;
    int endurance_cycles = 0;
    double mse = 0.0;
    double max_abs_err = 0.0;
    double snr_db = 0.0;
    double snr_adc_theory_db = 0.0;
};

TwoLayerResult run_two_layer(const std::string& name, const Config& cfg,
                             const std::vector<std::vector<double>>& W1,
                             const std::vector<std::vector<double>>& W2,
                             const std::vector<float>& digital_inputs,
                             const TwoLayerOptions& opt);

}  // namespace volt
