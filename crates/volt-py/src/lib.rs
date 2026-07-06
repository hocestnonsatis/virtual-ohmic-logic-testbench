use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use volt_core::{
    cell_current, circuit_transfer_f32, load_config_from_json, parse_circuit_model, run_two_layer,
    CircuitModel, Config, CrossbarArray, IvModel, SimulatedAdc, SimulatedDac, TwoLayerOptions,
};

#[pyclass(name = "IvModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyIvModel {
    Linear = 0,
    PowerLaw = 1,
    SoftSaturation = 2,
}

impl From<IvModel> for PyIvModel {
    fn from(m: IvModel) -> Self {
        match m {
            IvModel::Linear => PyIvModel::Linear,
            IvModel::PowerLaw => PyIvModel::PowerLaw,
            IvModel::SoftSaturation => PyIvModel::SoftSaturation,
        }
    }
}

impl From<PyIvModel> for IvModel {
    fn from(m: PyIvModel) -> Self {
        match m {
            PyIvModel::Linear => IvModel::Linear,
            PyIvModel::PowerLaw => IvModel::PowerLaw,
            PyIvModel::SoftSaturation => IvModel::SoftSaturation,
        }
    }
}

#[pyclass(name = "CircuitModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyCircuitModel {
    PassThrough = 0,
    DiodeRectifier = 1,
    TunableSigmoid = 2,
}

impl From<CircuitModel> for PyCircuitModel {
    fn from(m: CircuitModel) -> Self {
        match m {
            CircuitModel::PassThrough => PyCircuitModel::PassThrough,
            CircuitModel::DiodeRectifier => PyCircuitModel::DiodeRectifier,
            CircuitModel::TunableSigmoid => PyCircuitModel::TunableSigmoid,
        }
    }
}

impl From<PyCircuitModel> for CircuitModel {
    fn from(m: PyCircuitModel) -> Self {
        match m {
            PyCircuitModel::PassThrough => CircuitModel::PassThrough,
            PyCircuitModel::DiodeRectifier => CircuitModel::DiodeRectifier,
            PyCircuitModel::TunableSigmoid => CircuitModel::TunableSigmoid,
        }
    }
}

#[pyclass(name = "Config")]
#[derive(Clone)]
struct PyConfig {
    inner: Config,
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: Config::default(),
        }
    }

    #[getter]
    #[pyo3(name = "G_min")]
    fn g_min(&self) -> f32 {
        self.inner.g_min
    }
    #[setter]
    fn set_g_min(&mut self, v: f32) {
        self.inner.g_min = v;
    }

    #[getter]
    fn g_max(&self) -> f32 {
        self.inner.g_max
    }
    #[setter]
    fn set_g_max(&mut self, v: f32) {
        self.inner.g_max = v;
    }

    #[getter]
    fn v_min(&self) -> f32 {
        self.inner.v_min
    }
    #[setter]
    fn set_v_min(&mut self, v: f32) {
        self.inner.v_min = v;
    }

    #[getter]
    fn v_max(&self) -> f32 {
        self.inner.v_max
    }
    #[setter]
    fn set_v_max(&mut self, v: f32) {
        self.inner.v_max = v;
    }

    #[getter]
    fn i_min(&self) -> f32 {
        self.inner.i_min
    }
    #[setter]
    fn set_i_min(&mut self, v: f32) {
        self.inner.i_min = v;
    }

    #[getter]
    fn i_range(&self) -> f32 {
        self.inner.i_range
    }
    #[setter]
    fn set_i_range(&mut self, v: f32) {
        self.inner.i_range = v;
    }

    #[getter]
    fn n_bits_adc(&self) -> i32 {
        self.inner.n_bits_adc
    }
    #[setter]
    fn set_n_bits_adc(&mut self, v: i32) {
        self.inner.n_bits_adc = v;
    }

    #[getter]
    fn noise_stddev(&self) -> f32 {
        self.inner.noise_stddev
    }
    #[setter]
    fn set_noise_stddev(&mut self, v: f32) {
        self.inner.noise_stddev = v;
    }

    #[getter]
    fn disturb_ratio(&self) -> f32 {
        self.inner.disturb_ratio
    }
    #[setter]
    fn set_disturb_ratio(&mut self, v: f32) {
        self.inner.disturb_ratio = v;
    }

    #[getter]
    fn disturb_alpha(&self) -> f32 {
        self.inner.disturb_alpha
    }
    #[setter]
    fn set_disturb_alpha(&mut self, v: f32) {
        self.inner.disturb_alpha = v;
    }

    #[getter]
    fn noise_seed(&self) -> u32 {
        self.inner.noise_seed
    }
    #[setter]
    fn set_noise_seed(&mut self, v: u32) {
        self.inner.noise_seed = v;
    }

    #[getter]
    fn activation_sigmoid_steepness(&self) -> f32 {
        self.inner.activation_sigmoid_steepness
    }
    #[setter]
    fn set_activation_sigmoid_steepness(&mut self, v: f32) {
        self.inner.activation_sigmoid_steepness = v;
    }

    #[getter]
    fn write_endurance_lambda(&self) -> f32 {
        self.inner.write_endurance_lambda
    }
    #[setter]
    fn set_write_endurance_lambda(&mut self, v: f32) {
        self.inner.write_endurance_lambda = v;
    }

    #[getter]
    fn iv_model(&self) -> PyIvModel {
        self.inner.iv_model.into()
    }
    #[setter]
    fn set_iv_model(&mut self, v: PyIvModel) {
        self.inner.iv_model = v.into();
    }

    #[getter]
    fn iv_exponent(&self) -> f32 {
        self.inner.iv_exponent
    }
    #[setter]
    fn set_iv_exponent(&mut self, v: f32) {
        self.inner.iv_exponent = v;
    }

    #[getter]
    fn iv_v_ref(&self) -> f32 {
        self.inner.iv_v_ref
    }
    #[setter]
    fn set_iv_v_ref(&mut self, v: f32) {
        self.inner.iv_v_ref = v;
    }

    #[getter]
    fn iv_v_sat(&self) -> f32 {
        self.inner.iv_v_sat
    }
    #[setter]
    fn set_iv_v_sat(&mut self, v: f32) {
        self.inner.iv_v_sat = v;
    }

    #[getter]
    fn interlayer_circuit(&self) -> PyCircuitModel {
        self.inner.interlayer_circuit.into()
    }
    #[setter]
    fn set_interlayer_circuit(&mut self, v: PyCircuitModel) {
        self.inner.interlayer_circuit = v.into();
    }

    #[getter]
    fn circuit_i_threshold(&self) -> f32 {
        self.inner.circuit_i_threshold
    }
    #[setter]
    fn set_circuit_i_threshold(&mut self, v: f32) {
        self.inner.circuit_i_threshold = v;
    }

    #[getter]
    fn circuit_steepness(&self) -> f32 {
        self.inner.circuit_steepness
    }
    #[setter]
    fn set_circuit_steepness(&mut self, v: f32) {
        self.inner.circuit_steepness = v;
    }
}

#[pyclass]
struct PyCrossbarArray {
    inner: CrossbarArray,
}

#[pymethods]
impl PyCrossbarArray {
    #[new]
    fn new(rows: i32, cols: i32, cfg: &PyConfig) -> Self {
        Self {
            inner: CrossbarArray::new(rows, cols, cfg.inner.clone()),
        }
    }

    fn load_weights(&mut self, weights: Vec<Vec<f32>>) -> PyResult<()> {
        self.inner
            .load_weights(&weights)
            .map_err(PyRuntimeError::new_err)
    }

    fn apply_voltage(&self, voltages: Vec<f32>) -> PyResult<Vec<f32>> {
        self.inner
            .apply_voltage(&voltages)
            .map_err(PyRuntimeError::new_err)
    }

    fn rows(&self) -> i32 {
        self.inner.rows()
    }

    fn cols(&self) -> i32 {
        self.inner.cols()
    }

    fn effective_g_max(&self) -> f32 {
        self.inner.effective_g_max()
    }
}

#[pyclass]
struct PySimulatedDac {
    inner: SimulatedDac,
}

#[pymethods]
impl PySimulatedDac {
    #[new]
    fn new(cfg: &PyConfig) -> Self {
        Self {
            inner: SimulatedDac::new(cfg.inner.clone()),
        }
    }

    fn convert(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.inner.convert(&inputs)
    }
}

#[pyclass]
struct PySimulatedAdc {
    inner: SimulatedAdc,
}

#[pymethods]
impl PySimulatedAdc {
    #[new]
    fn new(cfg: &PyConfig) -> Self {
        Self {
            inner: SimulatedAdc::new(cfg.inner.clone()),
        }
    }

    fn quantize(&self, current: f32) -> i32 {
        self.inner.quantize(current)
    }

    fn reconstruct(&self, level: i32) -> f32 {
        self.inner.reconstruct(level)
    }

    fn level_to_dac_normalized(&self, level: i32) -> f32 {
        self.inner.level_to_dac_normalized(level)
    }

    fn max_level(&self) -> i32 {
        self.inner.max_level()
    }
}

#[pyfunction]
fn load_config_json(text: &str, mut base: PyConfig) -> PyResult<PyConfig> {
    load_config_from_json(text, &mut base.inner).map_err(PyRuntimeError::new_err)?;
    Ok(base)
}

#[pyfunction]
fn forward(
    weights: Vec<Vec<f32>>,
    inputs: Vec<f32>,
    cfg: &PyConfig,
) -> PyResult<(Vec<f32>, Vec<i32>)> {
    let rows = weights.len();
    if rows < 1 {
        return Err(PyRuntimeError::new_err("weights must be non-empty"));
    }
    let cols = weights[0].len();
    let dac = SimulatedDac::new(cfg.inner.clone());
    let mut cb = CrossbarArray::new(rows as i32, cols as i32, cfg.inner.clone());
    cb.load_weights(&weights).map_err(PyRuntimeError::new_err)?;
    let voltages = dac.convert(&inputs);
    let currents = cb.apply_voltage(&voltages).map_err(PyRuntimeError::new_err)?;
    let adc = SimulatedAdc::new(cfg.inner.clone());
    let levels: Vec<i32> = currents.iter().map(|&c| adc.quantize(c)).collect();
    Ok((currents, levels))
}

#[pyfunction]
#[pyo3(signature = (w1, w2, inputs, cfg, interlayer="pass_through"))]
fn two_layer_forward(
    w1: Vec<Vec<f32>>,
    w2: Vec<Vec<f32>>,
    inputs: Vec<f32>,
    cfg: &PyConfig,
    interlayer: &str,
) -> PyResult<PyObject> {
    let circuit = parse_circuit_model(interlayer)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown interlayer circuit: {interlayer}")))?;

    let w1d: Vec<Vec<f64>> = w1
        .iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect();
    let w2d: Vec<Vec<f64>> = w2
        .iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect();

    let mut opt = TwoLayerOptions::default();
    opt.interlayer_circuit = circuit;

    let r = run_two_layer("py_two_layer", &cfg.inner, &w1d, &w2d, &inputs, &opt)
        .map_err(PyRuntimeError::new_err)?;

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("mse", r.mse)?;
        dict.set_item("snr_db", r.snr_db)?;
        dict.set_item("max_abs_err", r.max_abs_err)?;
        Ok(dict.into())
    })
}

#[pyfunction]
#[pyo3(name = "cell_current")]
fn py_cell_current(v: f32, g: f32, model: PyIvModel, cfg: &PyConfig) -> f32 {
    cell_current(v, g, model.into(), &cfg.inner)
}

#[pyfunction]
#[pyo3(name = "circuit_transfer")]
fn py_circuit_transfer(i_in: f32, model: PyCircuitModel, cfg: &PyConfig) -> f32 {
    circuit_transfer_f32(i_in, model.into(), &cfg.inner)
}

#[pymodule]
fn volt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIvModel>()?;
    m.add_class::<PyCircuitModel>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyCrossbarArray>()?;
    m.add_class::<PySimulatedDac>()?;
    m.add_class::<PySimulatedAdc>()?;
    m.add_function(wrap_pyfunction!(load_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(two_layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(py_cell_current, m)?)?;
    m.add_function(wrap_pyfunction!(py_circuit_transfer, m)?)?;
    Ok(())
}
