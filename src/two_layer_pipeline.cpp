#include "two_layer_pipeline.hpp"

#include "activation_circuit.hpp"
#include "adc.hpp"
#include "crossbar.hpp"
#include "dac.hpp"
#include "noise.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace volt {

namespace {

std::vector<double> reference_currents(const std::vector<float>& voltages,
                                       const std::vector<std::vector<double>>& W,
                                       double G_max) {
    const int rows = static_cast<int>(voltages.size());
    if (rows < 1 || static_cast<int>(W.size()) != rows || W[0].empty()) {
        throw std::invalid_argument("reference_currents: dimension mismatch");
    }
    const int cols = static_cast<int>(W[0].size());
    for (int i = 0; i < rows; ++i) {
        if (static_cast<int>(W[static_cast<std::size_t>(i)].size()) != cols) {
            throw std::invalid_argument("reference_currents: ragged weight matrix");
        }
    }
    std::vector<double> I(static_cast<std::size_t>(cols), 0.0);
    for (int j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i) {
            sum += static_cast<double>(voltages[static_cast<std::size_t>(i)]) *
                   W[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * G_max;
        }
        I[static_cast<std::size_t>(j)] = sum;
    }
    return I;
}

void apply_crossbar_physics(CrossbarArray& crossbar, ThermalNoiseInjector& thermal,
                            ReadDisturbSimulator& disturb, const TwoLayerOptions& opt,
                            int active_row) {
    if (opt.write_endurance_cycles > 0) {
        WriteEnduranceSimulator wend(crossbar.config());
        wend.apply_write_cycles(crossbar, opt.write_endurance_cycles);
    }
    if (opt.use_persistent_noise) {
        thermal.inject_persistent(crossbar);
    }
    for (int c = 0; c < opt.disturb_cycles; ++c) {
        disturb.apply_disturb(crossbar, active_row, crossbar.config().V_max);
    }
}

std::vector<float> apply_voltage_with_noise(CrossbarArray& crossbar,
                                            const std::vector<float>& voltages,
                                            ThermalNoiseInjector& thermal, bool transient) {
    const int rows = crossbar.rows();
    const int cols = crossbar.cols();
    if (!transient) {
        return crossbar.apply_voltage(voltages);
    }
    std::vector<std::vector<float>> Gp(static_cast<std::size_t>(rows),
                                        std::vector<float>(static_cast<std::size_t>(cols)));
    std::vector<std::vector<float>> Gn(static_cast<std::size_t>(rows),
                                        std::vector<float>(static_cast<std::size_t>(cols)));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Gp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = crossbar.g_pos_at(i, j);
            Gn[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = crossbar.g_neg_at(i, j);
        }
    }
    std::vector<float> flat;
    flat.reserve(static_cast<std::size_t>(2 * rows * cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat.push_back(Gp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat.push_back(Gn[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    thermal.inject_transient(flat);
    std::size_t k = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Gp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = flat[k++];
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Gn[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = flat[k++];
        }
    }
    return crossbar.apply_voltage(voltages, Gp, Gn);
}

float interlayer_process(float I, const TwoLayerOptions& opt, const Config& cfg) {
    float out = circuit_transfer(I, opt.interlayer_circuit, cfg);
    if (opt.interlayer_activation != Activation::Identity) {
        out = apply_activation(out, opt.interlayer_activation, cfg);
    }
    return out;
}

double interlayer_process(double I, const TwoLayerOptions& opt, const Config& cfg) {
    double out = circuit_transfer(I, opt.interlayer_circuit, cfg);
    if (opt.interlayer_activation != Activation::Identity) {
        out = apply_activation(out, opt.interlayer_activation, cfg);
    }
    return out;
}

}  // namespace

TwoLayerResult run_two_layer(const std::string& name, const Config& cfg,
                             const std::vector<std::vector<double>>& W1,
                             const std::vector<std::vector<double>>& W2,
                             const std::vector<float>& digital_inputs,
                             const TwoLayerOptions& opt) {
    const int in_dim = static_cast<int>(W1.size());
    if (in_dim < 1 || W1[0].empty()) {
        throw std::invalid_argument("run_two_layer: W1 must be non-empty");
    }
    const int hidden_dim = static_cast<int>(W1[0].size());
    for (int i = 0; i < in_dim; ++i) {
        if (static_cast<int>(W1[static_cast<std::size_t>(i)].size()) != hidden_dim) {
            throw std::invalid_argument("run_two_layer: W1 is ragged");
        }
    }
    if (static_cast<int>(W2.size()) != hidden_dim || W2[0].empty()) {
        throw std::invalid_argument("run_two_layer: W2 row count must match W1 cols");
    }
    const int out_dim = static_cast<int>(W2[0].size());
    for (int i = 0; i < hidden_dim; ++i) {
        if (static_cast<int>(W2[static_cast<std::size_t>(i)].size()) != out_dim) {
            throw std::invalid_argument("run_two_layer: W2 is ragged");
        }
    }
    if (static_cast<int>(digital_inputs.size()) != in_dim) {
        throw std::invalid_argument("run_two_layer: input length must equal W1 rows");
    }

    SimulatedDAC dac(cfg);
    CrossbarArray crossbar1(in_dim, hidden_dim, cfg);
    CrossbarArray crossbar2(hidden_dim, out_dim, cfg);
    SimulatedADC adc(cfg);
    ThermalNoiseInjector thermal(cfg);
    ReadDisturbSimulator disturb(cfg);

    std::vector<std::vector<float>> Wf1(static_cast<std::size_t>(in_dim),
                                        std::vector<float>(static_cast<std::size_t>(hidden_dim)));
    std::vector<std::vector<float>> Wf2(static_cast<std::size_t>(hidden_dim),
                                        std::vector<float>(static_cast<std::size_t>(out_dim)));
    for (int i = 0; i < in_dim; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            Wf1[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(W1[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    for (int i = 0; i < hidden_dim; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            Wf2[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(W2[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    crossbar1.load_weights(Wf1);
    crossbar2.load_weights(Wf2);

    const int active_row1 = in_dim <= 1 ? 0 : std::min(2, in_dim - 1);
    const int active_row2 = hidden_dim <= 1 ? 0 : std::min(2, hidden_dim - 1);
    apply_crossbar_physics(crossbar1, thermal, disturb, opt, active_row1);
    apply_crossbar_physics(crossbar2, thermal, disturb, opt, active_row2);

    std::vector<float> voltages1 = dac.convert(digital_inputs);
    std::vector<float> currents1 =
        apply_voltage_with_noise(crossbar1, voltages1, thermal, opt.use_transient_noise);

    std::vector<float> hidden_processed(static_cast<std::size_t>(hidden_dim));
    for (int j = 0; j < hidden_dim; ++j) {
        hidden_processed[static_cast<std::size_t>(j)] = interlayer_process(
            currents1[static_cast<std::size_t>(j)], opt, cfg);
    }

    std::vector<float> dac_inputs_l2(static_cast<std::size_t>(hidden_dim));
    for (int j = 0; j < hidden_dim; ++j) {
        const int lv = adc.quantize(hidden_processed[static_cast<std::size_t>(j)]);
        dac_inputs_l2[static_cast<std::size_t>(j)] = adc.level_to_dac_normalized(lv);
    }
    std::vector<float> voltages2 = dac.convert(dac_inputs_l2);
    std::vector<float> currents2 =
        apply_voltage_with_noise(crossbar2, voltages2, thermal, opt.use_transient_noise);

    std::vector<double> I1_ref =
        reference_currents(voltages1, W1, static_cast<double>(crossbar1.effective_g_max()));

    std::vector<float> hidden_ref_q(static_cast<std::size_t>(hidden_dim));
    for (int j = 0; j < hidden_dim; ++j) {
        const double processed =
            interlayer_process(I1_ref[static_cast<std::size_t>(j)], opt, cfg);
        const int lv = adc.quantize(static_cast<float>(processed));
        hidden_ref_q[static_cast<std::size_t>(j)] = adc.level_to_dac_normalized(lv);
    }
    std::vector<float> voltages2_ref = dac.convert(hidden_ref_q);
    std::vector<double> I2_ref = reference_currents(
        voltages2_ref, W2, static_cast<double>(crossbar2.effective_g_max()));

    double mse = 0.0;
    double max_abs = 0.0;
    double mean_ref_sq = 0.0;
    for (int j = 0; j < out_dim; ++j) {
        const int level = adc.quantize(currents2[static_cast<std::size_t>(j)]);
        const float recon = adc.reconstruct(level);
        const double ref = I2_ref[static_cast<std::size_t>(j)];
        const double err = static_cast<double>(recon) - ref;
        mse += err * err;
        max_abs = std::max(max_abs, std::abs(err));
        mean_ref_sq += ref * ref;
    }
    mse /= static_cast<double>(out_dim);
    mean_ref_sq /= static_cast<double>(out_dim);

    double snr_db = 0.0;
    if (mse > 1e-30) {
        snr_db = 10.0 * std::log10(mean_ref_sq / mse);
    } else {
        snr_db = 200.0;
    }

    const double Ps =
        static_cast<double>(cfg.I_range) * static_cast<double>(cfg.I_range) / 8.0;
    const double delta =
        static_cast<double>(cfg.I_range) / static_cast<double>(adc.max_level());
    const double Pq = delta * delta / 12.0;
    const double snr_adc_theory_db = 10.0 * std::log10(Ps / Pq);

    TwoLayerResult r;
    r.name = name;
    r.n_bits = cfg.n_bits_adc;
    r.noise_stddev = cfg.noise_stddev;
    r.disturb_cycles = opt.disturb_cycles;
    r.endurance_cycles = opt.write_endurance_cycles;
    r.mse = mse;
    r.max_abs_err = max_abs;
    r.snr_db = snr_db;
    r.snr_adc_theory_db = snr_adc_theory_db;
    return r;
}

}  // namespace volt
