#include "activation.hpp"
#include "adc.hpp"
#include "config.hpp"
#include "config_json.hpp"
#include "crossbar.hpp"
#include "weights_csv.hpp"
#include "dac.hpp"
#include "noise.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

void assert_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void assert_near(float a, float b, float eps, const char* msg) {
    if (std::abs(a - b) > eps) {
        std::cerr << "FAIL: " << msg << " (" << a << " vs " << b << ")\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    volt::Config cfg;

    // --- DAC ---
    {
        volt::SimulatedDAC dac(cfg);
        auto z = dac.convert(std::vector<float>{0.0f});
        assert_near(z[0], cfg.V_min, 1e-6f, "DAC 0 -> V_min");
        auto o = dac.convert(std::vector<float>{1.0f});
        assert_near(o[0], cfg.V_max, 1e-5f, "DAC 1 -> V_max");
        auto h = dac.convert(std::vector<float>{0.5f});
        assert_near(h[0], (cfg.V_min + cfg.V_max) * 0.5f, 1e-5f, "DAC 0.5 mid");
        std::ostringstream buf;
        std::streambuf* old = std::cerr.rdbuf(buf.rdbuf());
        auto bad = dac.convert(std::vector<float>{1.5f});
        std::cerr.rdbuf(old);
        assert_near(bad[0], cfg.V_max, 1e-5f, "DAC 1.5 clamped to V_max");
        assert_true(buf.str().find("warning") != std::string::npos, "DAC out-of-range warning");

        std::vector<std::uint8_t> raw = {0, 255, 128};
        auto vr = dac.convert(raw);
        assert_near(vr[0], cfg.V_min, 1e-5f, "DAC uint8 0");
        assert_near(vr[1], cfg.V_max, 1e-5f, "DAC uint8 255");
        float exp128 = cfg.V_min + (128.0f / 255.0f) * (cfg.V_max - cfg.V_min);
        assert_near(vr[2], exp128, 1e-5f, "DAC uint8 128");
    }

    // --- ADC (bipolar window [I_min, I_min + I_range]) ---
    {
        cfg.n_bits_adc = 8;
        volt::SimulatedADC adc8(cfg);
        float mid = cfg.I_min + cfg.I_range * 0.5f;
        int l8 = adc8.quantize(mid);
        assert_true(std::abs(l8 - 127) <= 1, "8-bit half range level ~127");

        cfg.n_bits_adc = 4;
        volt::SimulatedADC adc4(cfg);
        int l4 = adc4.quantize(mid);
        assert_true(std::abs(l4 - 7) <= 1, "4-bit half range level ~7");

        cfg.n_bits_adc = 8;
        volt::SimulatedADC adc(cfg);
        assert_true(adc.quantize(cfg.I_min) == 0, "ADC bottom of window");
        int lmax = adc.quantize(cfg.I_min + cfg.I_range);
        assert_true(lmax == adc.max_level(), "ADC full scale");
        int lhi = adc.quantize(cfg.I_min + cfg.I_range * 100.0f);
        assert_true(lhi == adc.max_level(), "ADC over range clamped");

        assert_near(adc.level_to_dac_normalized(0), 0.0f, 1e-7f, "ADC L0 -> DAC 0");
        assert_near(adc.level_to_dac_normalized(adc.max_level()), 1.0f, 1e-6f,
                    "ADC Lmax -> DAC 1");
        const int mid_lv = adc.max_level() / 2;
        const float exp_norm =
            static_cast<float>(mid_lv) / static_cast<float>(adc.max_level());
        assert_near(adc.level_to_dac_normalized(mid_lv), exp_norm, 1e-6f,
                    "ADC mid level -> normalized");
    }

    // --- Analog activation (post-MAC I_net) ---
    {
        volt::Config c;
        assert_near(volt::apply_activation(-1.0e-5f, volt::Activation::ReLU, c), 0.0f, 1e-12f,
                    "ReLU negative -> 0");
        assert_near(volt::apply_activation(3.0e-5f, volt::Activation::ReLU, c), 3.0e-5f, 1e-12f,
                    "ReLU positive passthrough");
        const float ys =
            volt::apply_activation(0.0f, volt::Activation::Sigmoid, c);
        assert_true(ys >= c.I_min && ys <= c.I_min + c.I_range, "Sigmoid in ADC window");
        const float yhi = volt::apply_activation(c.I_min + c.I_range * 10.0f, volt::Activation::Sigmoid,
                                                 c);
        assert_near(yhi, c.I_min + c.I_range, 1e-5f * c.I_range, "Sigmoid large I near top");
    }

    // --- Crossbar weights & identity ---
    {
        volt::CrossbarArray xb(2, 2, cfg);
        std::vector<std::vector<float>> Wp = {{1.0f, 0.0f}, {0.0f, 1.0f}};
        xb.load_weights(Wp);
        assert_near(xb.g_pos_at(0, 0), cfg.G_max, 1e-5f * cfg.G_max, "w=1 G_pos");
        assert_near(xb.g_neg_at(0, 0), 0.0f, 1e-5f * cfg.G_max, "w=1 G_neg");
        assert_near(xb.g_pos_at(1, 1), cfg.G_max, 1e-5f * cfg.G_max, "w=1 diag");
        assert_near(xb.g_neg_at(1, 1), 0.0f, 1e-5f * cfg.G_max, "w=1 G_neg diag");

        volt::CrossbarArray xbn(2, 2, cfg);
        std::vector<std::vector<float>> Wn = {{-1.0f, 0.0f}, {0.0f, -1.0f}};
        xbn.load_weights(Wn);
        assert_near(xbn.g_pos_at(0, 0), 0.0f, 1e-5f * cfg.G_max, "w=-1 G_pos");
        assert_near(xbn.g_neg_at(0, 0), cfg.G_max, 1e-5f * cfg.G_max, "w=-1 G_neg");

        volt::CrossbarArray xbz(2, 2, cfg);
        std::vector<std::vector<float>> Wz = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        xbz.load_weights(Wz);
        assert_near(xbz.g_pos_at(0, 0), cfg.G_max * 0.5f, 1e-5f * cfg.G_max, "w=0 G_pos");
        assert_near(xbz.g_neg_at(0, 0), cfg.G_max * 0.5f, 1e-5f * cfg.G_max, "w=0 G_neg");

        volt::CrossbarArray xr(2, 2, cfg);
        std::vector<std::vector<float>> Wr = {{0.3f, -0.7f}, {0.1f, 0.9f}};
        xr.load_weights(Wr);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                assert_near(xr.get_effective_weight(i, j), Wr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                            1e-4f, "effective weight recovery");
            }
        }

        std::vector<float> V = {0.3f, 0.7f};
        auto I = xb.apply_voltage(V);
        assert_near(I[0], 0.3f * cfg.G_max, 1e-4f * cfg.G_max, "identity I0");
        assert_near(I[1], 0.7f * cfg.G_max, 1e-4f * cfg.G_max, "identity I1");
    }

    // --- Thermal noise ---
    {
        volt::Config c0;
        c0.noise_stddev = 0.0f;
        volt::ThermalNoiseInjector n0(c0);
        std::vector<float> g = {1.0f, 2.0f, 3.0f};
        std::vector<float> gcopy = g;
        n0.inject_transient(gcopy);
        assert_near(gcopy[0], g[0], 1e-7f, "noise 0 stddev unchanged");

        volt::Config c1;
        c1.noise_stddev = 0.01f;
        c1.noise_seed = 12345;
        volt::ThermalNoiseInjector a(c1);
        volt::ThermalNoiseInjector b(c1);
        std::vector<float> v1 = {cfg.G_max * 0.5f, cfg.G_max * 0.25f};
        std::vector<float> v2 = v1;
        a.inject_transient(v1);
        b.inject_transient(v2);
        assert_near(v1[0], v2[0], 1e-6f, "fixed seed identical noise");
        assert_near(v1[1], v2[1], 1e-6f, "fixed seed identical noise 2");

        volt::CrossbarArray arr(2, 2, cfg);
        std::vector<std::vector<float>> W = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        arr.load_weights(W);
        c1.noise_stddev = 0.01f;
        volt::ThermalNoiseInjector np(c1);
        np.inject_persistent(arr);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                assert_true(arr.g_pos_at(i, j) >= 0.0f && arr.g_pos_at(i, j) <= cfg.G_max,
                            "G_pos in range");
                assert_true(arr.g_neg_at(i, j) >= 0.0f && arr.g_neg_at(i, j) <= cfg.G_max,
                            "G_neg in range");
            }
        }
    }

    // --- Read disturb ---
    {
        volt::Config cd;
        cd.disturb_alpha = 1e-5f;
        cd.disturb_ratio = 0.03f;
        volt::CrossbarArray arr(4, 4, cd);
        std::vector<std::vector<float>> Wz(4, std::vector<float>(4, 0.0f));
        arr.load_weights(Wz);
        float g0 = arr.g_pos_at(1, 0);
        volt::ReadDisturbSimulator ds(cd);
        float Vapp = 0.01f;
        const int cycles = 1000;
        for (int k = 0; k < cycles; ++k) {
            ds.apply_disturb(arr, 2, Vapp);
        }
        float g1 = arr.g_pos_at(1, 0);
        float expected_delta = static_cast<float>(cycles) * cd.disturb_alpha * (Vapp * cd.disturb_ratio);
        float actual_delta = g1 - g0;
        assert_near(actual_delta, expected_delta, 1e-7f, "read disturb cumulative delta");
    }

    // --- Write endurance (uniform G scale vs. cycles) ---
    {
        volt::Config ce;
        ce.write_endurance_lambda = 1e-5f;
        volt::CrossbarArray ar(2, 2, ce);
        std::vector<std::vector<float>> ww = {{1.0f, 0.0f}, {0.0f, 1.0f}};
        ar.load_weights(ww);
        const float g0 = ar.effective_g_max();
        volt::WriteEnduranceSimulator we(ce);
        we.apply_write_cycles(ar, 100000);
        const float g1 = ar.effective_g_max();
        assert_true(g1 < g0, "endurance reduces effective G_max");
        const float scale = std::exp(-1e-5f * 100000.0f);
        assert_near(g1, ce.G_max * scale, 1e-5f * ce.G_max, "exp(-lambda * cycles) scale");

        volt::Config c0;
        c0.write_endurance_lambda = 0.0f;
        volt::WriteEnduranceSimulator w0(c0);
        volt::CrossbarArray ar2(2, 2, c0);
        ar2.load_weights(ww);
        const float before = ar2.effective_g_max();
        w0.apply_write_cycles(ar2, 999999);
        assert_near(ar2.effective_g_max(), before, 1e-12f, "lambda=0 no endurance");
    }

    // --- JSON config (subset parser) ---
    {
        volt::Config c;
        std::string err;
        const std::string_view j = R"({"G_max":2e-4,"noise_seed":99,"n_bits_adc":6})";
        assert_true(volt::load_config_from_json(j, c, err), "JSON parse ok");
        assert_true(err.empty(), "JSON no error string");
        assert_near(c.G_max, 2e-4f, 1e-12f, "JSON G_max");
        assert_true(c.noise_seed == 99u, "JSON noise_seed");
        assert_true(c.n_bits_adc == 6, "JSON n_bits_adc");

        volt::Config c2;
        assert_true(volt::load_config_from_json(R"({})", c2, err), "empty JSON object");
        assert_near(c2.G_max, volt::Config{}.G_max, 1e-12f, "empty JSON preserves defaults");
    }

    // --- CSV weights ---
    {
        const char* tmp = "volt_test_weights.csv";
        std::ofstream f(tmp);
        assert_true(f.good(), "temp csv open");
        f << "# identity\n";
        f << "1,0,0,0\n0,1,0,0\n0,0,1,0\n0,0,0,1\n";
        f.close();
        std::vector<std::vector<double>> W;
        std::string err;
        assert_true(volt::load_weights_csv_file(tmp, W, err), err.c_str());
        std::remove(tmp);
        assert_near(static_cast<float>(W[0][0]), 1.0f, 1e-9f, "CSV W(0,0)");
        assert_near(static_cast<float>(W[1][1]), 1.0f, 1e-9f, "CSV W(1,1)");
    }
    {
        const char* tmp = "volt_test_weights_3.csv";
        std::ofstream f(tmp);
        f << "1,0,0\n0,-1,0\n0,0,0.5\n";
        f.close();
        std::vector<std::vector<double>> W;
        std::string err;
        assert_true(volt::load_weights_csv_file(tmp, W, err), err.c_str());
        std::remove(tmp);
        assert_true(static_cast<int>(W.size()) == 3, "3x3 rows");
        assert_near(static_cast<float>(W[2][2]), 0.5f, 1e-9f, "3x3 corner");
    }
    // --- CSV inputs ---
    {
        const char* tmp = "volt_test_inputs.csv";
        std::ofstream f(tmp);
        f << "0.2\n0.5\n# c\n0.8\n";
        f.close();
        std::vector<float> in;
        std::string err;
        assert_true(volt::load_inputs_csv_file(tmp, 3, in, err), err.c_str());
        std::remove(tmp);
        assert_near(in[0], 0.2f, 1e-9f, "in[0]");
        assert_near(in[2], 0.8f, 1e-9f, "in[2]");
    }

    std::cout << "test_core: all checks passed\n";
    return 0;
}
