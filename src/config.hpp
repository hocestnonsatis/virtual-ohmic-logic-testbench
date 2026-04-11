#pragma once

namespace volt {

struct Config {
    float G_min = 1e-6f;           // Siemens
    float G_max = 1e-4f;           // Siemens
    float V_min = 0.1f;            // Volts
    float V_max = 1.5f;            // Volts
    /// Bottom of ADC input window (amperes). Top is `I_min + I_range`.
    /// Calibrated to the default 4×4 demo matrix and inputs so I_net stays inside the window.
    float I_min = -6.02e-5f;
    /// Full-scale span: `I_step = I_range / (2^n_bits - 1)`. Span matches peak-to-peak I_net
    /// for the demo (~−6.02 µA … +9.14 µA → 1.516×10⁻⁴ A) with negligible margin.
    float I_range = 1.516e-4f;
    int n_bits_adc = 8;
    float noise_stddev = 0.0f;
    float disturb_ratio = 0.03f;
    float disturb_alpha = 1e-5f;
    unsigned int noise_seed = 42;
};

}  // namespace volt
