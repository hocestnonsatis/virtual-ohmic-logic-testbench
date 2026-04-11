#include "adc.hpp"
#include "config.hpp"
#include "crossbar.hpp"
#include "dac.hpp"
#include "noise.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using volt::Config;
using volt::CrossbarArray;
using volt::ReadDisturbSimulator;
using volt::SimulatedADC;
using volt::SimulatedDAC;
using volt::ThermalNoiseInjector;

struct ScenarioResult {
    std::string name;
    int n_bits;
    float noise_stddev;
    int disturb_cycles;
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
                            bool use_persistent_noise, bool print_currents) {
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

    std::vector<double> I_ref = reference_currents(voltages, W_double, static_cast<double>(cfg.G_max));

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
    r.mse = mse;
    r.max_abs_err = max_abs;
    r.snr_db = snr_db;
    r.snr_adc_theory_db = snr_adc_theory_db;
    return r;
}

}  // namespace

int main() {
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

    std::cout << std::fixed << std::setprecision(8);
    for (const auto& r : results) {
        std::cout << "=== " << r.name << " ===\n";
        std::cout << std::scientific << std::setprecision(6) << "  MSE: " << r.mse << "\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  Max absolute error (A): " << r.max_abs_err << "\n";
        std::cout << "  SNR (dB, measured mean(I_ref^2)/MSE): " << r.snr_db << "\n";
        std::cout << "  SQNR (dB, ADC full-scale sine vs Delta^2/12): " << r.snr_adc_theory_db
                  << "\n\n";
    }

    std::ofstream csv("results.csv");
    csv << "scenario,n_bits,noise_stddev,disturb_cycles,mse,max_abs_err,snr_measured_db,snr_adc_"
           "theory_db\n";
    csv << std::fixed << std::setprecision(12);
    for (const auto& r : results) {
        csv << r.name << ',' << r.n_bits << ',' << r.noise_stddev << ',' << r.disturb_cycles << ','
            << r.mse << ',' << r.max_abs_err << ',' << r.snr_db << ',' << r.snr_adc_theory_db << '\n';
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
