use crate::config::Config;
use crate::iv_model::cell_current;

pub struct CrossbarArray {
    rows: i32,
    cols: i32,
    cfg: Config,
    g_max_effective: f32,
    pub(crate) g_pos: Vec<Vec<f32>>,
    pub(crate) g_neg: Vec<Vec<f32>>,
    pub(crate) g_pos_baseline: Vec<Vec<f32>>,
    pub(crate) g_neg_baseline: Vec<Vec<f32>>,
}

impl CrossbarArray {
    pub fn new(rows: i32, cols: i32, cfg: Config) -> Self {
        let g_pos = vec![vec![cfg.g_min; cols as usize]; rows as usize];
        let g_neg = g_pos.clone();
        Self {
            rows,
            cols,
            g_max_effective: cfg.g_max,
            g_pos_baseline: g_pos.clone(),
            g_neg_baseline: g_neg.clone(),
            g_pos,
            g_neg,
            cfg,
        }
    }

    pub fn load_weights(&mut self, weights: &[Vec<f32>]) -> Result<(), String> {
        if weights.len() as i32 != self.rows {
            return Err("CrossbarArray::load_weights: row count mismatch".into());
        }
        for row in weights {
            if row.len() as i32 != self.cols {
                return Err("CrossbarArray::load_weights: column count mismatch".into());
            }
        }
        for (i, row) in weights.iter().enumerate() {
            for (j, &w_in) in row.iter().enumerate() {
                let mut w = w_in;
                if w < -1.0 || w > 1.0 {
                    eprintln!(
                        "[CrossbarArray] warning: weight {w} outside [-1,1]; clamping"
                    );
                    w = w.clamp(-1.0, 1.0);
                }
                self.g_pos[i][j] = ((w + 1.0) / 2.0) * self.cfg.g_max;
                self.g_neg[i][j] = ((1.0 - w) / 2.0) * self.cfg.g_max;
            }
        }
        self.g_max_effective = self.cfg.g_max;
        self.g_pos_baseline = self.g_pos.clone();
        self.g_neg_baseline = self.g_neg.clone();
        Ok(())
    }

    pub fn get_effective_weight(&self, i: i32, j: i32) -> f32 {
        let gp = self.g_pos[i as usize][j as usize];
        let gn = self.g_neg[i as usize][j as usize];
        (gp - gn) / self.cfg.g_max
    }

    pub fn apply_voltage(&self, voltages: &[f32]) -> Result<Vec<f32>, String> {
        self.apply_voltage_with_g(voltages, &self.g_pos, &self.g_neg)
    }

    pub fn apply_voltage_with_g(
        &self,
        voltages: &[f32],
        g_pos: &[Vec<f32>],
        g_neg: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        if voltages.len() as i32 != self.rows {
            return Err("CrossbarArray::apply_voltage: voltage count != rows".into());
        }
        if g_pos.len() as i32 != self.rows || g_neg.len() as i32 != self.rows {
            return Err("CrossbarArray::apply_voltage: conductance row mismatch".into());
        }
        for i in 0..self.rows as usize {
            if g_pos[i].len() as i32 != self.cols || g_neg[i].len() as i32 != self.cols {
                return Err("CrossbarArray::apply_voltage: conductance col mismatch".into());
            }
        }

        let v_row: Vec<f32> = voltages
            .iter()
            .map(|&v| v.clamp(self.cfg.v_min, self.cfg.v_max))
            .collect();

        let mut i_pos = vec![0.0f32; self.cols as usize];
        let mut i_neg = vec![0.0f32; self.cols as usize];

        for j in 0..self.cols as usize {
            for i in 0..self.rows as usize {
                let v = v_row[i];
                i_pos[j] += cell_current(v, g_pos[i][j], self.cfg.iv_model, &self.cfg);
                i_neg[j] += cell_current(v, g_neg[i][j], self.cfg.iv_model, &self.cfg);
            }
        }

        Ok(i_pos
            .iter()
            .zip(i_neg.iter())
            .map(|(&p, &n)| p - n)
            .collect())
    }

    pub fn rows(&self) -> i32 {
        self.rows
    }

    pub fn cols(&self) -> i32 {
        self.cols
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn g_pos_at(&self, i: i32, j: i32) -> f32 {
        self.g_pos[i as usize][j as usize]
    }

    pub fn g_neg_at(&self, i: i32, j: i32) -> f32 {
        self.g_neg[i as usize][j as usize]
    }

    pub fn effective_g_max(&self) -> f32 {
        self.g_max_effective
    }

    pub(crate) fn apply_uniform_conductance_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.0, 1.0);
        self.g_max_effective = self.cfg.g_max * scale;
        for i in 0..self.rows as usize {
            for j in 0..self.cols as usize {
                self.g_pos[i][j] = (self.g_pos[i][j] * scale).clamp(0.0, self.g_max_effective);
                self.g_neg[i][j] = (self.g_neg[i][j] * scale).clamp(0.0, self.g_max_effective);
            }
        }
        self.g_pos_baseline = self.g_pos.clone();
        self.g_neg_baseline = self.g_neg.clone();
    }
}
