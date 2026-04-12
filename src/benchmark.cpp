#include "benchmark.hpp"

#include "crossbar.hpp"
#include "dac.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace volt {

namespace {

float deterministic_weight(int i, int j) {
    const unsigned ui = static_cast<unsigned>(i) * 2654435761u;
    const unsigned uj = static_cast<unsigned>(j) * 2246822519u;
    const unsigned u = (ui ^ uj ^ (ui << 3U)) % 2001u;
    float w = static_cast<float>(static_cast<int>(u)) / 1000.0f - 1.0f;
    return std::clamp(w, -1.0f, 1.0f);
}

}  // namespace

void run_benchmark_suite(const Config& cfg) {
    SimulatedDAC dac(cfg);

    constexpr int k_sizes[] = {4, 8, 16, 32, 64, 128};
    constexpr auto k_target = std::chrono::milliseconds(40);

    std::cout << "[benchmark] n×n crossbar MAC (Ohm/KCL); timing steady-state forwards\n";

    std::ofstream csv("benchmark.csv");
    csv << "n,repetitions,total_s,forwards_per_s,gmac_per_s,ns_per_forward\n";
    csv << std::fixed << std::setprecision(9);

    for (int n : k_sizes) {
        CrossbarArray cb(n, n, cfg);
        std::vector<std::vector<float>> wf(static_cast<std::size_t>(n),
                                           std::vector<float>(static_cast<std::size_t>(n)));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                wf[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    deterministic_weight(i, j);
            }
        }
        cb.load_weights(wf);

        std::vector<float> in(static_cast<std::size_t>(n), 0.5f);
        for (int i = 0; i < n; ++i) {
            in[static_cast<std::size_t>(i)] =
                0.15f + 0.7f * static_cast<float>(i) / static_cast<float>(std::max(1, n - 1));
        }
        std::vector<float> voltages = dac.convert(in);

        float sink = 0.0f;
        for (int w = 0; w < 8; ++w) {
            std::vector<float> cur = cb.apply_voltage(voltages);
            for (float x : cur) {
                sink += x;
            }
        }
        (void)sink;

        std::int64_t reps = 0;
        const auto t_begin = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t_begin < k_target) {
            for (int k = 0; k < 64; ++k) {
                std::vector<float> cur = cb.apply_voltage(voltages);
                for (float x : cur) {
                    sink += x;
                }
                ++reps;
            }
        }
        const auto t_end = std::chrono::steady_clock::now();
        const double total_s =
            std::chrono::duration<double>(t_end - t_begin).count();
        if (total_s <= 0.0 || reps <= 0) {
            continue;
        }
        const double fwd_per_s = static_cast<double>(reps) / total_s;
        const double mac_per_fwd = static_cast<double>(n) * static_cast<double>(n);
        const double gmac_per_s = mac_per_fwd * fwd_per_s / 1e9;
        const double ns_per_fwd = 1e9 / fwd_per_s;

        std::cout << "  n=" << n << "  reps=" << reps << "  " << fwd_per_s << " fwd/s  "
                  << gmac_per_s << " GMAC/s\n";
        csv << n << ',' << reps << ',' << total_s << ',' << fwd_per_s << ',' << gmac_per_s << ','
            << ns_per_fwd << '\n';
    }

    std::cout << "[benchmark] wrote benchmark.csv (cwd)\n";
}

}  // namespace volt
