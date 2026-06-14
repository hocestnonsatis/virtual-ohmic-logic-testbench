#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "activation_circuit.hpp"
#include "adc.hpp"
#include "config.hpp"
#include "config_json.hpp"
#include "crossbar.hpp"
#include "dac.hpp"
#include "iv_model.hpp"
#include "two_layer_pipeline.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

volt::CircuitModel parse_interlayer(const std::string& name) {
    volt::CircuitModel m;
    if (volt::parse_circuit_model(name, m)) {
        return m;
    }
    throw std::invalid_argument("unknown interlayer circuit: " + name);
}

}  // namespace

PYBIND11_MODULE(volt, m) {
    m.doc() = "VOLT — Virtual Ohmic Logic Testbench Python bindings";

    py::enum_<volt::IvModel>(m, "IvModel")
        .value("Linear", volt::IvModel::Linear)
        .value("PowerLaw", volt::IvModel::PowerLaw)
        .value("SoftSaturation", volt::IvModel::SoftSaturation);

    py::enum_<volt::CircuitModel>(m, "CircuitModel")
        .value("PassThrough", volt::CircuitModel::PassThrough)
        .value("DiodeRectifier", volt::CircuitModel::DiodeRectifier)
        .value("TunableSigmoid", volt::CircuitModel::TunableSigmoid);

    py::class_<volt::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("G_min", &volt::Config::G_min)
        .def_readwrite("G_max", &volt::Config::G_max)
        .def_readwrite("V_min", &volt::Config::V_min)
        .def_readwrite("V_max", &volt::Config::V_max)
        .def_readwrite("I_min", &volt::Config::I_min)
        .def_readwrite("I_range", &volt::Config::I_range)
        .def_readwrite("n_bits_adc", &volt::Config::n_bits_adc)
        .def_readwrite("noise_stddev", &volt::Config::noise_stddev)
        .def_readwrite("disturb_ratio", &volt::Config::disturb_ratio)
        .def_readwrite("disturb_alpha", &volt::Config::disturb_alpha)
        .def_readwrite("noise_seed", &volt::Config::noise_seed)
        .def_readwrite("activation_sigmoid_steepness", &volt::Config::activation_sigmoid_steepness)
        .def_readwrite("write_endurance_lambda", &volt::Config::write_endurance_lambda)
        .def_readwrite("iv_model", &volt::Config::iv_model)
        .def_readwrite("iv_exponent", &volt::Config::iv_exponent)
        .def_readwrite("iv_v_ref", &volt::Config::iv_v_ref)
        .def_readwrite("iv_v_sat", &volt::Config::iv_v_sat)
        .def_readwrite("interlayer_circuit", &volt::Config::interlayer_circuit)
        .def_readwrite("circuit_i_threshold", &volt::Config::circuit_i_threshold)
        .def_readwrite("circuit_steepness", &volt::Config::circuit_steepness);

    py::class_<volt::CrossbarArray>(m, "CrossbarArray")
        .def(py::init<int, int, const volt::Config&>())
        .def("load_weights", &volt::CrossbarArray::load_weights)
        .def("apply_voltage",
             py::overload_cast<const std::vector<float>&>(
                 &volt::CrossbarArray::apply_voltage))
        .def("rows", &volt::CrossbarArray::rows)
        .def("cols", &volt::CrossbarArray::cols)
        .def("effective_g_max", &volt::CrossbarArray::effective_g_max);

    py::class_<volt::SimulatedDAC>(m, "SimulatedDAC")
        .def(py::init<const volt::Config&>())
        .def("convert", py::overload_cast<const std::vector<float>&>(
                             &volt::SimulatedDAC::convert));

    py::class_<volt::SimulatedADC>(m, "SimulatedADC")
        .def(py::init<const volt::Config&>())
        .def("quantize", &volt::SimulatedADC::quantize)
        .def("reconstruct", &volt::SimulatedADC::reconstruct)
        .def("level_to_dac_normalized", &volt::SimulatedADC::level_to_dac_normalized)
        .def("max_level", &volt::SimulatedADC::max_level);

    m.def("load_config_json", [](const std::string& text, volt::Config base) {
        std::string err;
        if (!volt::load_config_from_json(text, base, err)) {
            throw std::runtime_error(err);
        }
        return base;
    });

    m.def("forward",
          [](const std::vector<std::vector<float>>& weights, const std::vector<float>& inputs,
             const volt::Config& cfg) {
              const int rows = static_cast<int>(weights.size());
              if (rows < 1) {
                  throw std::invalid_argument("weights must be non-empty");
              }
              const int cols = static_cast<int>(weights[0].size());
              volt::SimulatedDAC dac(cfg);
              volt::CrossbarArray cb(rows, cols, cfg);
              cb.load_weights(weights);
              auto voltages = dac.convert(inputs);
              auto currents = cb.apply_voltage(voltages);
              volt::SimulatedADC adc(cfg);
              std::vector<int> levels(currents.size());
              for (std::size_t j = 0; j < currents.size(); ++j) {
                  levels[j] = adc.quantize(currents[j]);
              }
              return py::make_tuple(currents, levels);
          });

    m.def("two_layer_forward",
          [](const std::vector<std::vector<float>>& W1,
             const std::vector<std::vector<float>>& W2, const std::vector<float>& inputs,
             const volt::Config& cfg, const std::string& interlayer) {
              std::vector<std::vector<double>> W1d(W1.size());
              for (std::size_t i = 0; i < W1.size(); ++i) {
                  W1d[i].resize(W1[i].size());
                  for (std::size_t j = 0; j < W1[i].size(); ++j) {
                      W1d[i][j] = static_cast<double>(W1[i][j]);
                  }
              }
              std::vector<std::vector<double>> W2d(W2.size());
              for (std::size_t i = 0; i < W2.size(); ++i) {
                  W2d[i].resize(W2[i].size());
                  for (std::size_t j = 0; j < W2[i].size(); ++j) {
                      W2d[i][j] = static_cast<double>(W2[i][j]);
                  }
              }
              volt::TwoLayerOptions opt;
              opt.interlayer_circuit = parse_interlayer(interlayer);
              auto r = volt::run_two_layer("py_two_layer", cfg, W1d, W2d, inputs, opt);
              return py::dict(py::arg("mse") = r.mse, py::arg("snr_db") = r.snr_db,
                              py::arg("max_abs_err") = r.max_abs_err);
          },
          py::arg("W1"), py::arg("W2"), py::arg("inputs"), py::arg("cfg"),
          py::arg("interlayer") = "pass_through");

    m.def("cell_current", &volt::cell_current);
    m.def("circuit_transfer",
          py::overload_cast<float, volt::CircuitModel, const volt::Config&>(
              &volt::circuit_transfer));
}
