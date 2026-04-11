#include "adc.hpp"
#include "config.hpp"
#include "crossbar.hpp"
#include "dac.hpp"
#include "noise.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void fail(const char* msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

double scenario_a_mse() {
    const std::vector<std::vector<double>> W = {
        {0.8, -0.3, 0.5, -0.1},
        {-0.6, 0.9, -0.2, 0.7},
        {0.1, -0.8, 0.4, -0.5},
        {0.3, 0.2, -0.9, 0.6},
    };
    const std::vector<float> inputs = {0.9f, 0.4f, 0.7f, 0.2f};

    volt::Config cfg;
    volt::SimulatedDAC dac(cfg);
    volt::CrossbarArray crossbar(4, 4, cfg);
    volt::SimulatedADC adc(cfg);

    std::vector<std::vector<float>> Wf(4, std::vector<float>(4));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Wf[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(W[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
        }
    }
    crossbar.load_weights(Wf);

    std::vector<float> voltages = dac.convert(inputs);

    std::vector<double> I_ref(4, 0.0);
    for (int j = 0; j < 4; ++j) {
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            sum += static_cast<double>(voltages[static_cast<std::size_t>(i)]) *
                   W[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                   static_cast<double>(cfg.G_max);
        }
        I_ref[static_cast<std::size_t>(j)] = sum;
    }

    std::vector<float> currents = crossbar.apply_voltage(voltages);

    double mse = 0.0;
    for (int j = 0; j < 4; ++j) {
        int level = adc.quantize(currents[static_cast<std::size_t>(j)]);
        if (level > adc.max_level()) {
            fail("ADC level exceeds max");
        }
        float recon = adc.reconstruct(level);
        double err = static_cast<double>(recon) - I_ref[static_cast<std::size_t>(j)];
        mse += err * err;
    }
    return mse / 4.0;
}

void assert_conductances_ok(volt::CrossbarArray& cb) {
    for (int i = 0; i < cb.rows(); ++i) {
        for (int j = 0; j < cb.cols(); ++j) {
            float gmax = cb.config().G_max;
            float gp = cb.g_pos_at(i, j);
            float gn = cb.g_neg_at(i, j);
            if (gp < 0.0f || gp > gmax || gn < 0.0f || gn > gmax) {
                fail("conductance out of [0, G_max]");
            }
        }
    }
}

}  // namespace

int main() {
    volt::Config cfg;
    volt::SimulatedADC adc(cfg);

    double mse = scenario_a_mse();
    if (mse >= 1e-6) {
        std::cerr << "MSE = " << mse << '\n';
        fail("Scenario A MSE must be < 1e-6");
    }

    volt::CrossbarArray cb(4, 4, cfg);
    std::vector<std::vector<float>> W = {
        {0.5f, -0.2f, 0.0f, 0.3f},
        {-0.1f, 0.4f, 0.6f, -0.9f},
        {0.2f, 0.2f, -0.5f, 0.1f},
        {0.0f, 0.8f, -0.3f, 0.4f},
    };
    cb.load_weights(W);
    assert_conductances_ok(cb);

    volt::ThermalNoiseInjector th(cfg);
    th.inject_persistent(cb);
    assert_conductances_ok(cb);

    volt::ReadDisturbSimulator rd(cfg);
    rd.apply_disturb(cb, 1, cfg.V_max);
    assert_conductances_ok(cb);

    for (float c = cfg.I_min - cfg.I_range * 0.1f; c <= cfg.I_min + cfg.I_range * 1.1f;
         c += cfg.I_range * 0.05f) {
        int lv = adc.quantize(c);
        if (lv < 0 || lv > adc.max_level()) {
            fail("ADC quantize out of range");
        }
    }

    std::cout << "test_equivalence: all checks passed\n";
    return 0;
}
