use crate::config::{Config, IvModel};

fn sign_v(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub fn cell_current(v: f32, g: f32, model: IvModel, cfg: &Config) -> f32 {
    match model {
        IvModel::Linear => g * v,
        IvModel::PowerLaw => {
            let v_ref = cfg.iv_v_ref.max(1e-12);
            let alpha = cfg.iv_exponent;
            let mag = (v.abs() / v_ref).powf(alpha);
            g * sign_v(v) * mag * v_ref
        }
        IvModel::SoftSaturation => {
            let v_sat = cfg.iv_v_sat.max(1e-12);
            g * v / (1.0 + v.abs() / v_sat)
        }
    }
}

pub fn parse_iv_model(name: &str) -> Option<IvModel> {
    match name.to_ascii_lowercase().as_str() {
        "linear" => Some(IvModel::Linear),
        "power_law" | "powerlaw" => Some(IvModel::PowerLaw),
        "soft_saturation" | "softsaturation" => Some(IvModel::SoftSaturation),
        _ => None,
    }
}

pub fn iv_model_name(model: IvModel) -> &'static str {
    match model {
        IvModel::Linear => "linear",
        IvModel::PowerLaw => "power_law",
        IvModel::SoftSaturation => "soft_saturation",
    }
}
