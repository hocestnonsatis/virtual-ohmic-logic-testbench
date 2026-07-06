use rand_distr::{Distribution, Normal};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::config::Config;
use crate::crossbar::CrossbarArray;

pub struct ThermalNoiseInjector {
    cfg: Config,
    rng: StdRng,
    dist: Normal<f32>,
}

impl ThermalNoiseInjector {
    pub fn new(cfg: Config) -> Self {
        let dist = Normal::new(0.0, cfg.noise_stddev).expect("invalid noise stddev");
        let rng = StdRng::seed_from_u64(cfg.noise_seed as u64);
        Self { cfg, rng, dist }
    }

    pub fn inject_transient(&mut self, conductances: &mut [f32]) {
        if self.cfg.noise_stddev <= 0.0 {
            return;
        }
        for g in conductances.iter_mut() {
            *g += self.dist.sample(&mut self.rng);
        }
    }

    pub fn inject_persistent(&mut self, array: &mut CrossbarArray) {
        if self.cfg.noise_stddev <= 0.0 {
            return;
        }
        let gmax = array.effective_g_max();
        let rows = array.rows();
        let cols = array.cols();
        for i in 0..rows {
            for j in 0..cols {
                let np = self.dist.sample(&mut self.rng);
                let nn = self.dist.sample(&mut self.rng);
                array.g_pos[i as usize][j as usize] =
                    (array.g_pos[i as usize][j as usize] + np).clamp(0.0, gmax);
                array.g_neg[i as usize][j as usize] =
                    (array.g_neg[i as usize][j as usize] + nn).clamp(0.0, gmax);
            }
        }
    }
}

pub struct ReadDisturbSimulator {
    cfg: Config,
}

impl ReadDisturbSimulator {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    pub fn apply_disturb(&self, array: &mut CrossbarArray, active_row: i32, v_applied: f32) {
        if active_row < 0 || active_row >= array.rows() {
            return;
        }
        let v_dis = v_applied * self.cfg.disturb_ratio;
        let delta = self.cfg.disturb_alpha * v_dis;
        let gmax = array.effective_g_max();
        for neighbor in [active_row - 1, active_row + 1] {
            if neighbor < 0 || neighbor >= array.rows() {
                continue;
            }
            for j in 0..array.cols() {
                let ni = neighbor as usize;
                let nj = j as usize;
                array.g_pos[ni][nj] = (array.g_pos[ni][nj] + delta).clamp(0.0, gmax);
                array.g_neg[ni][nj] = (array.g_neg[ni][nj] + delta).clamp(0.0, gmax);
            }
        }
    }

    pub fn log_drift_report(&self, array: &CrossbarArray) {
        let n = (array.rows() * array.cols()) as i32;
        if n == 0 {
            println!("[ReadDisturb] drift report: empty array");
            return;
        }
        let mut sum_shift = 0.0f64;
        let mut max_shift = 0.0f64;
        for i in 0..array.rows() {
            for j in 0..array.cols() {
                let gp0 = array.g_pos_baseline[i as usize][j as usize];
                let gn0 = array.g_neg_baseline[i as usize][j as usize];
                let gp = array.g_pos[i as usize][j as usize];
                let gn = array.g_neg[i as usize][j as usize];
                let shift_pos = (gp as f64 - gp0 as f64).abs();
                let shift_neg = (gn as f64 - gn0 as f64).abs();
                let cell_shift = shift_pos.max(shift_neg);
                sum_shift += cell_shift;
                max_shift = max_shift.max(cell_shift);
            }
        }
        let avg = sum_shift / n as f64;
        println!(
            "[ReadDisturb] drift: avg conductance shift = {avg} S, max = {max_shift} S"
        );
    }
}

pub struct WriteEnduranceSimulator {
    cfg: Config,
}

impl WriteEnduranceSimulator {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    pub fn apply_write_cycles(&self, array: &mut CrossbarArray, cycles: i32) {
        if cycles <= 0 || self.cfg.write_endurance_lambda <= 0.0 {
            return;
        }
        let scale = (-self.cfg.write_endurance_lambda * cycles as f32).exp();
        array.apply_uniform_conductance_scale(scale);
    }
}
