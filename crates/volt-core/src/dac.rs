use crate::config::Config;

pub struct SimulatedDac {
    cfg: Config,
}

impl SimulatedDac {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    pub fn v_min(&self) -> f32 {
        self.cfg.v_min
    }

    pub fn v_max(&self) -> f32 {
        self.cfg.v_max
    }

    pub fn convert(&self, inputs: &[f32]) -> Vec<f32> {
        inputs
            .iter()
            .map(|&x| {
                let clamped = if x < 0.0 || x > 1.0 {
                    eprintln!(
                        "[SimulatedDAC] warning: input {x} outside [0,1]; clamping to [0,1]"
                    );
                    x.clamp(0.0, 1.0)
                } else {
                    x
                };
                self.cfg.v_min + clamped * (self.cfg.v_max - self.cfg.v_min)
            })
            .collect()
    }

    pub fn convert_u8(&self, inputs: &[u8]) -> Vec<f32> {
        let normalized: Vec<f32> = inputs.iter().map(|&b| b as f32 / 255.0).collect();
        self.convert(&normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dac_endpoints() {
        let cfg = Config::default();
        let dac = SimulatedDac::new(cfg.clone());
        assert!((dac.convert(&[0.0])[0] - cfg.v_min).abs() < 1e-6);
        assert!((dac.convert(&[1.0])[0] - cfg.v_max).abs() < 1e-5);
        let mid = (cfg.v_min + cfg.v_max) * 0.5;
        assert!((dac.convert(&[0.5])[0] - mid).abs() < 1e-5);
    }
}
