#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IvModel {
    #[default]
    Linear,
    PowerLaw,
    SoftSaturation,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CircuitModel {
    #[default]
    PassThrough,
    DiodeRectifier,
    TunableSigmoid,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub g_min: f32,
    pub g_max: f32,
    pub v_min: f32,
    pub v_max: f32,
    pub i_min: f32,
    pub i_range: f32,
    pub n_bits_adc: i32,
    pub noise_stddev: f32,
    pub disturb_ratio: f32,
    pub disturb_alpha: f32,
    pub noise_seed: u32,
    pub activation_sigmoid_steepness: f32,
    pub write_endurance_lambda: f32,
    pub iv_model: IvModel,
    pub iv_exponent: f32,
    pub iv_v_ref: f32,
    pub iv_v_sat: f32,
    pub interlayer_circuit: CircuitModel,
    pub circuit_i_threshold: f32,
    pub circuit_steepness: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            g_min: 1e-6,
            g_max: 1e-4,
            v_min: 0.1,
            v_max: 1.5,
            i_min: -6.02e-5,
            i_range: 1.516e-4,
            n_bits_adc: 8,
            noise_stddev: 0.0,
            disturb_ratio: 0.03,
            disturb_alpha: 1e-5,
            noise_seed: 42,
            activation_sigmoid_steepness: 6.0,
            write_endurance_lambda: 0.0,
            iv_model: IvModel::Linear,
            iv_exponent: 1.0,
            iv_v_ref: 1.0,
            iv_v_sat: 1.5,
            interlayer_circuit: CircuitModel::PassThrough,
            circuit_i_threshold: 0.0,
            circuit_steepness: 6.0,
        }
    }
}
