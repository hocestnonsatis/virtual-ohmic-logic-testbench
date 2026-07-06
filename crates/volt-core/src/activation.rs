use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Activation {
    #[default]
    Identity,
    ReLU,
    Sigmoid,
}

pub fn apply_activation_f32(i: f32, a: Activation, cfg: &Config) -> f32 {
    match a {
        Activation::Identity => i,
        Activation::ReLU => i.max(0.0),
        Activation::Sigmoid => {
            let mid = cfg.i_min + 0.5 * cfg.i_range;
            let scale = cfg.i_range * 0.25 + 1e-30;
            let x = (i - mid) / scale * cfg.activation_sigmoid_steepness;
            let s = 1.0 / (1.0 + (-x).exp());
            cfg.i_min + cfg.i_range * s
        }
    }
}

pub fn apply_activation_f64(i: f64, a: Activation, cfg: &Config) -> f64 {
    match a {
        Activation::Identity => i,
        Activation::ReLU => i.max(0.0),
        Activation::Sigmoid => {
            let mid = cfg.i_min as f64 + 0.5 * cfg.i_range as f64;
            let scale = cfg.i_range as f64 * 0.25 + 1e-30;
            let x = (i - mid) / scale * cfg.activation_sigmoid_steepness as f64;
            let s = 1.0 / (1.0 + (-x).exp());
            cfg.i_min as f64 + cfg.i_range as f64 * s
        }
    }
}
