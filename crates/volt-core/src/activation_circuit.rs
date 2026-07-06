use crate::config::{CircuitModel, Config};

fn sigmoid_circuit_f32(i_in: f32, cfg: &Config) -> f32 {
    let mid = cfg.i_min + 0.5 * cfg.i_range;
    let scale = cfg.i_range * 0.25 + 1e-30;
    let x = (i_in - mid) / scale * cfg.circuit_steepness;
    let s = 1.0 / (1.0 + (-x).exp());
    cfg.i_min + cfg.i_range * s
}

fn sigmoid_circuit_f64(i_in: f64, cfg: &Config) -> f64 {
    let mid = cfg.i_min as f64 + 0.5 * cfg.i_range as f64;
    let scale = cfg.i_range as f64 * 0.25 + 1e-30;
    let x = (i_in - mid) / scale * cfg.circuit_steepness as f64;
    let s = 1.0 / (1.0 + (-x).exp());
    cfg.i_min as f64 + cfg.i_range as f64 * s
}

pub fn circuit_transfer_f32(i_in: f32, model: CircuitModel, cfg: &Config) -> f32 {
    match model {
        CircuitModel::PassThrough => i_in,
        CircuitModel::DiodeRectifier => (i_in - cfg.circuit_i_threshold).max(0.0),
        CircuitModel::TunableSigmoid => sigmoid_circuit_f32(i_in, cfg),
    }
}

pub fn circuit_transfer_f64(i_in: f64, model: CircuitModel, cfg: &Config) -> f64 {
    match model {
        CircuitModel::PassThrough => i_in,
        CircuitModel::DiodeRectifier => {
            (i_in - cfg.circuit_i_threshold as f64).max(0.0)
        }
        CircuitModel::TunableSigmoid => sigmoid_circuit_f64(i_in, cfg),
    }
}

pub fn parse_circuit_model(name: &str) -> Option<CircuitModel> {
    match name.to_ascii_lowercase().as_str() {
        "pass_through" | "passthrough" | "identity" => Some(CircuitModel::PassThrough),
        "diode_rectifier" | "diode" | "relu" => Some(CircuitModel::DiodeRectifier),
        "tunable_sigmoid" | "sigmoid" => Some(CircuitModel::TunableSigmoid),
        _ => None,
    }
}

pub fn circuit_model_name(model: CircuitModel) -> &'static str {
    match model {
        CircuitModel::PassThrough => "pass_through",
        CircuitModel::DiodeRectifier => "diode_rectifier",
        CircuitModel::TunableSigmoid => "tunable_sigmoid",
    }
}
