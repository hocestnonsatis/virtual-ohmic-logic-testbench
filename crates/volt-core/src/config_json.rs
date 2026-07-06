use serde_json::Value;

use crate::activation_circuit::parse_circuit_model;
use crate::config::Config;
use crate::iv_model::parse_iv_model;

fn apply_numeric_key(key: &str, v: f64, cfg: &mut Config) {
    match key {
        "G_min" => cfg.g_min = v as f32,
        "G_max" => cfg.g_max = v as f32,
        "V_min" => cfg.v_min = v as f32,
        "V_max" => cfg.v_max = v as f32,
        "I_min" => cfg.i_min = v as f32,
        "I_range" => cfg.i_range = v as f32,
        "n_bits_adc" => cfg.n_bits_adc = v.round() as i32,
        "noise_stddev" => cfg.noise_stddev = v as f32,
        "disturb_ratio" => cfg.disturb_ratio = v as f32,
        "disturb_alpha" => cfg.disturb_alpha = v as f32,
        "noise_seed" => cfg.noise_seed = v.round() as u32,
        "activation_sigmoid_steepness" => cfg.activation_sigmoid_steepness = v as f32,
        "write_endurance_lambda" => cfg.write_endurance_lambda = v as f32,
        "iv_exponent" => cfg.iv_exponent = v as f32,
        "iv_v_ref" => cfg.iv_v_ref = v as f32,
        "iv_v_sat" => cfg.iv_v_sat = v as f32,
        "circuit_i_threshold" => cfg.circuit_i_threshold = v as f32,
        "circuit_steepness" => cfg.circuit_steepness = v as f32,
        _ => {}
    }
}

fn apply_string_key(key: &str, value: &str, cfg: &mut Config) {
    match key {
        "iv_model" => {
            if let Some(m) = parse_iv_model(value) {
                cfg.iv_model = m;
            }
        }
        "interlayer_circuit" => {
            if let Some(m) = parse_circuit_model(value) {
                cfg.interlayer_circuit = m;
            }
        }
        _ => {}
    }
}

fn apply_value(key: &str, value: &Value, cfg: &mut Config) -> Result<(), String> {
    match value {
        Value::String(s) => {
            apply_string_key(key, s, cfg);
            Ok(())
        }
        Value::Number(n) => {
            if let Some(v) = n.as_f64() {
                apply_numeric_key(key, v, cfg);
                Ok(())
            } else {
                Err(format!(
                    "JSON: expected number or string for key \"{key}\""
                ))
            }
        }
        _ => Err(format!(
            "JSON: expected number or string for key \"{key}\""
        )),
    }
}

pub fn load_config_from_json(text: &str, base: &mut Config) -> Result<(), String> {
    let v: Value =
        serde_json::from_str(text).map_err(|e| format!("JSON parse error: {e}"))?;
    let obj = v
        .as_object()
        .ok_or_else(|| "JSON: expected '{'".to_string())?;
    for (key, value) in obj {
        apply_value(key, value, base)?;
    }
    Ok(())
}

pub fn load_config_from_json_file(path: &str, base: &mut Config) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| format!("cannot open config file: {path}"))?;
    load_config_from_json(&text, base)
}
