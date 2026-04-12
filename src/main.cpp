#include "activation.hpp"
#include "adc.hpp"
#include "benchmark.hpp"
#include "config.hpp"
#include "crossbar.hpp"
#include "dac.hpp"
#include "noise.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using volt::Activation;
using volt::Config;
using volt::CrossbarArray;
using volt::ReadDisturbSimulator;
using volt::SimulatedADC;
using volt::SimulatedDAC;
using volt::ThermalNoiseInjector;
using volt::WriteEnduranceSimulator;

struct ScenarioResult {
    std::string name;
    int n_bits;
    float noise_stddev;
    int disturb_cycles;
    int endurance_cycles;
    double mse;
    double max_abs_err;
    /// Measured: mean(I_ref²) / MSE (reconstruction vs reference current).
    double snr_db;
    /// Classical ADC SQNR for full-scale sine vs uniform Δ²/12 (depends only on I_range, n_bits).
    double snr_adc_theory_db;
};

std::vector<double> reference_currents(const std::vector<float>& voltages,
                                         const std::vector<std::vector<double>>& W,
                                         double G_max) {
    const int n = static_cast<int>(voltages.size());
    std::vector<double> I(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += static_cast<double>(voltages[static_cast<std::size_t>(i)]) *
                   W[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * G_max;
        }
        I[static_cast<std::size_t>(j)] = sum;
    }
    return I;
}

ScenarioResult run_scenario(const std::string& name, Config cfg, int disturb_cycles,
                            const std::vector<std::vector<double>>& W_double,
                            const std::vector<float>& digital_inputs, bool use_transient_noise,
                            bool use_persistent_noise, bool print_currents,
                            Activation activation = Activation::Identity,
                            int write_endurance_cycles = 0) {
    SimulatedDAC dac(cfg);
    CrossbarArray crossbar(4, 4, cfg);
    SimulatedADC adc(cfg);
    ThermalNoiseInjector thermal(cfg);
    ReadDisturbSimulator disturb(cfg);

    std::vector<std::vector<float>> Wf(4, std::vector<float>(4));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Wf[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(W_double[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    crossbar.load_weights(Wf);

    if (write_endurance_cycles > 0) {
        WriteEnduranceSimulator wend(cfg);
        wend.apply_write_cycles(crossbar, write_endurance_cycles);
    }

    if (use_persistent_noise) {
        thermal.inject_persistent(crossbar);
    }

    for (int c = 0; c < disturb_cycles; ++c) {
        disturb.apply_disturb(crossbar, 2, cfg.V_max);
    }

    std::vector<float> voltages = dac.convert(digital_inputs);

    std::vector<std::vector<float>> Gp(4, std::vector<float>(4));
    std::vector<std::vector<float>> Gn(4, std::vector<float>(4));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Gp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = crossbar.g_pos_at(i, j);
            Gn[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = crossbar.g_neg_at(i, j);
        }
    }

    std::vector<float> currents;
    if (use_transient_noise) {
        std::vector<float> flat;
        flat.reserve(32);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                flat.push_back(Gp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                flat.push_back(Gn[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
        }
        thermal.inject_transient(flat);
        std::size_t k = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Gp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = flat[k++];
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Gn[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = flat[k++];
            }
        }
        currents = crossbar.apply_voltage(voltages, Gp, Gn);
    } else {
        currents = crossbar.apply_voltage(voltages);
    }

    if (activation != Activation::Identity) {
        for (int j = 0; j < 4; ++j) {
            currents[static_cast<std::size_t>(j)] = volt::apply_activation(
                currents[static_cast<std::size_t>(j)], activation, cfg);
        }
    }

    if (print_currents) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "[Scenario A] raw I_net before ADC (A): ";
        for (int j = 0; j < 4; ++j) {
            if (j > 0) {
                std::cout << ", ";
            }
            std::cout << currents[static_cast<std::size_t>(j)];
        }
        std::cout << "\n";
        std::cout << std::fixed;
    }

    std::vector<double> I_ref = reference_currents(
        voltages, W_double, static_cast<double>(crossbar.effective_g_max()));
    if (activation != Activation::Identity) {
        for (int j = 0; j < 4; ++j) {
            I_ref[static_cast<std::size_t>(j)] = volt::apply_activation(
                I_ref[static_cast<std::size_t>(j)], activation, cfg);
        }
    }

    double mse = 0.0;
    double max_abs = 0.0;
    double mean_ref_sq = 0.0;
    const int n = 4;
    for (int j = 0; j < n; ++j) {
        int level = adc.quantize(currents[static_cast<std::size_t>(j)]);
        float recon = adc.reconstruct(level);
        double ref = I_ref[static_cast<std::size_t>(j)];
        double err = static_cast<double>(recon) - ref;
        mse += err * err;
        max_abs = std::max(max_abs, std::abs(err));
        mean_ref_sq += ref * ref;
    }
    mse /= static_cast<double>(n);
    mean_ref_sq /= static_cast<double>(n);

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

    ScenarioResult r;
    r.name = name;
    r.n_bits = cfg.n_bits_adc;
    r.noise_stddev = cfg.noise_stddev;
    r.disturb_cycles = disturb_cycles;
    r.endurance_cycles = write_endurance_cycles;
    r.mse = mse;
    r.max_abs_err = max_abs;
    r.snr_db = snr_db;
    r.snr_adc_theory_db = snr_adc_theory_db;
    return r;
}

/// Two layers: layer-1 currents are quantized by the ADC; codes map to [0,1] and drive the
/// second-layer DAC. Reference uses continuous (I₁ − I_min) / I_range → DAC (no L1 quant).
ScenarioResult run_two_layer_scenario(const std::string& name, Config cfg,
                                      const std::vector<std::vector<double>>& W1,
                                      const std::vector<std::vector<double>>& W2,
                                      const std::vector<float>& digital_inputs) {
    SimulatedDAC dac(cfg);
    CrossbarArray crossbar1(4, 4, cfg);
    CrossbarArray crossbar2(4, 4, cfg);
    SimulatedADC adc(cfg);

    std::vector<std::vector<float>> Wf1(4, std::vector<float>(4));
    std::vector<std::vector<float>> Wf2(4, std::vector<float>(4));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Wf1[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(W1[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            Wf2[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(W2[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    crossbar1.load_weights(Wf1);
    crossbar2.load_weights(Wf2);

    std::vector<float> voltages1 = dac.convert(digital_inputs);
    std::vector<float> currents1 = crossbar1.apply_voltage(voltages1);

    std::vector<float> dac_inputs_l2(4);
    for (int j = 0; j < 4; ++j) {
        const int lv = adc.quantize(currents1[static_cast<std::size_t>(j)]);
        dac_inputs_l2[static_cast<std::size_t>(j)] = adc.level_to_dac_normalized(lv);
    }
    std::vector<float> voltages2 = dac.convert(dac_inputs_l2);
    std::vector<float> currents2 = crossbar2.apply_voltage(voltages2);

    std::vector<double> I1_ref = reference_currents(voltages1, W1, static_cast<double>(cfg.G_max));
    std::vector<float> norms_ideal(4);
    for (int j = 0; j < 4; ++j) {
        double t = (I1_ref[static_cast<std::size_t>(j)] - static_cast<double>(cfg.I_min)) /
                   static_cast<double>(cfg.I_range);
        t = std::clamp(t, 0.0, 1.0);
        norms_ideal[static_cast<std::size_t>(j)] = static_cast<float>(t);
    }
    std::vector<float> voltages2_ref = dac.convert(norms_ideal);
    std::vector<double> I2_ref =
        reference_currents(voltages2_ref, W2, static_cast<double>(cfg.G_max));

    double mse = 0.0;
    double max_abs = 0.0;
    double mean_ref_sq = 0.0;
    const int n = 4;
    for (int j = 0; j < n; ++j) {
        const int level = adc.quantize(currents2[static_cast<std::size_t>(j)]);
        const float recon = adc.reconstruct(level);
        const double ref = I2_ref[static_cast<std::size_t>(j)];
        const double err = static_cast<double>(recon) - ref;
        mse += err * err;
        max_abs = std::max(max_abs, std::abs(err));
        mean_ref_sq += ref * ref;
    }
    mse /= static_cast<double>(n);
    mean_ref_sq /= static_cast<double>(n);

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

    ScenarioResult r;
    r.name = name;
    r.n_bits = cfg.n_bits_adc;
    r.noise_stddev = cfg.noise_stddev;
    r.disturb_cycles = 0;
    r.endurance_cycles = 0;
    r.mse = mse;
    r.max_abs_err = max_abs;
    r.snr_db = snr_db;
    r.snr_adc_theory_db = snr_adc_theory_db;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--benchmark") == 0) {
        volt::run_benchmark_suite(volt::Config{});
        return 0;
    }

    const std::vector<std::vector<double>> W = {
        {0.8, -0.3, 0.5, -0.1},
        {-0.6, 0.9, -0.2, 0.7},
        {0.1, -0.8, 0.4, -0.5},
        {0.3, 0.2, -0.9, 0.6},
    };
    const std::vector<float> inputs = {0.9f, 0.4f, 0.7f, 0.2f};

    std::vector<ScenarioResult> results;

    {
        Config cfg;
        results.push_back(run_scenario("A_ideal", cfg, 0, W, inputs, false, false, true));
    }
    {
        Config cfg;
        cfg.n_bits_adc = 4;
        results.push_back(run_scenario("B_low_adc", cfg, 0, W, inputs, false, false, false));
    }
    {
        Config cfg;
        cfg.noise_stddev = 0.005f * cfg.G_max;
        results.push_back(run_scenario("C_thermal", cfg, 0, W, inputs, true, false, false));
    }
    {
        Config cfg;
        results.push_back(run_scenario("D_read_disturb", cfg, 1000, W, inputs, false, false, false));
    }
    {
        Config cfg;
        cfg.n_bits_adc = 4;
        cfg.noise_stddev = 0.005f * cfg.G_max;
        results.push_back(run_scenario("E_combined", cfg, 1000, W, inputs, true, true, false));
    }
    {
        Config cfg;
        // Second layer scaled so I_net stays inside default I_min / I_range window.
        const std::vector<std::vector<double>> W2 = {
            {0.5, 0.0, 0.0, 0.0},
            {0.0, 0.5, 0.0, 0.0},
            {0.0, 0.0, 0.5, 0.0},
            {0.0, 0.0, 0.0, 0.5},
        };
        results.push_back(run_two_layer_scenario("F_multilayer", cfg, W, W2, inputs));
    }
    {
        Config cfg;
        results.push_back(run_scenario("G_relu", cfg, 0, W, inputs, false, false, false,
                                       Activation::ReLU));
    }
    {
        Config cfg;
        results.push_back(run_scenario("H_sigmoid", cfg, 0, W, inputs, false, false, false,
                                       Activation::Sigmoid));
    }
    {
        Config cfg;
        cfg.write_endurance_lambda = 1e-5f;
        results.push_back(run_scenario("I_write_endurance", cfg, 0, W, inputs, false, false, false,
                                       Activation::Identity, 80000));
    }

    std::cout << std::fixed << std::setprecision(8);
    for (const auto& r : results) {
        std::cout << "=== " << r.name << " ===\n";
        std::cout << std::scientific << std::setprecision(6) << "  MSE: " << r.mse << "\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  Max absolute error (A): " << r.max_abs_err << "\n";
        std::cout << "  SNR (dB, measured mean(I_ref^2)/MSE): " << r.snr_db << "\n";
        std::cout << "  SQNR (dB, ADC full-scale sine vs Delta^2/12): " << r.snr_adc_theory_db
                  << "\n";
        if (r.endurance_cycles > 0) {
            std::cout << "  Write endurance cycles (modeled): " << r.endurance_cycles << "\n";
        }
        std::cout << "\n";
    }

    std::ofstream csv("results.csv");
    csv << "scenario,n_bits,noise_stddev,disturb_cycles,endurance_cycles,mse,max_abs_err,snr_"
           "measured_db,snr_adc_theory_db\n";
    csv << std::fixed << std::setprecision(12);
    for (const auto& r : results) {
        csv << r.name << ',' << r.n_bits << ',' << r.noise_stddev << ',' << r.disturb_cycles << ','
            << r.endurance_cycles << ',' << r.mse << ',' << r.max_abs_err << ',' << r.snr_db << ','
            << r.snr_adc_theory_db << '\n';
    }

    {
        Config cfg;
        CrossbarArray cb(4, 4, cfg);
        std::vector<std::vector<float>> Wf(4, std::vector<float>(4));
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Wf[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    static_cast<float>(W[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
        }
        cb.load_weights(Wf);
        ReadDisturbSimulator ds(cfg);
        for (int c = 0; c < 1000; ++c) {
            ds.apply_disturb(cb, 2, cfg.V_max);
        }
        ds.log_drift_report(cb);
    }

    return 0;
}
