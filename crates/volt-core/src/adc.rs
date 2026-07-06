use crate::config::Config;

pub struct SimulatedAdc {
    cfg: Config,
}

impl SimulatedAdc {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    pub fn max_level(&self) -> i32 {
        let n = self.cfg.n_bits_adc;
        if n < 1 {
            return 0;
        }
        (1 << n) - 1
    }

    pub fn i_step(&self) -> f32 {
        let denom = self.max_level();
        if denom <= 0 {
            return self.cfg.i_range;
        }
        self.cfg.i_range / denom as f32
    }

    pub fn quantize(&self, current: f32) -> i32 {
        let step = self.i_step();
        if step <= 0.0 {
            return 0;
        }
        let shifted = current - self.cfg.i_min;
        let raw = (shifted / step).floor() as i32;
        raw.clamp(0, self.max_level())
    }

    pub fn reconstruct(&self, level: i32) -> f32 {
        let level = level.clamp(0, self.max_level());
        self.cfg.i_min + level as f32 * self.i_step()
    }

    pub fn level_to_dac_normalized(&self, level: i32) -> f32 {
        let mx = self.max_level();
        if mx <= 0 {
            return 0.0;
        }
        let level = level.clamp(0, mx);
        level as f32 / mx as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adc_bipolar_window() {
        let mut cfg = Config::default();
        cfg.n_bits_adc = 8;
        let adc8 = SimulatedAdc::new(cfg.clone());
        let mid = cfg.i_min + cfg.i_range * 0.5;
        let l8 = adc8.quantize(mid);
        assert!((l8 - 127).abs() <= 1);

        cfg.n_bits_adc = 4;
        let adc4 = SimulatedAdc::new(cfg.clone());
        let l4 = adc4.quantize(mid);
        assert!((l4 - 7).abs() <= 1);
    }
}
