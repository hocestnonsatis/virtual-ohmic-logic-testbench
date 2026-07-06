use crate::activation::{apply_activation_f32, apply_activation_f64, Activation};
use crate::activation_circuit::{circuit_transfer_f32, circuit_transfer_f64};
use crate::adc::SimulatedAdc;
use crate::config::{CircuitModel, Config};
use crate::crossbar::CrossbarArray;
use crate::dac::SimulatedDac;
use crate::noise::{ReadDisturbSimulator, ThermalNoiseInjector, WriteEnduranceSimulator};

#[derive(Debug, Clone)]
pub struct TwoLayerOptions {
    pub interlayer_activation: Activation,
    pub interlayer_circuit: CircuitModel,
    pub use_transient_noise: bool,
    pub use_persistent_noise: bool,
    pub disturb_cycles: i32,
    pub write_endurance_cycles: i32,
}

impl Default for TwoLayerOptions {
    fn default() -> Self {
        Self {
            interlayer_activation: Activation::Identity,
            interlayer_circuit: CircuitModel::PassThrough,
            use_transient_noise: false,
            use_persistent_noise: false,
            disturb_cycles: 0,
            write_endurance_cycles: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwoLayerResult {
    pub name: String,
    pub n_bits: i32,
    pub noise_stddev: f32,
    pub disturb_cycles: i32,
    pub endurance_cycles: i32,
    pub mse: f64,
    pub max_abs_err: f64,
    pub snr_db: f64,
    pub snr_adc_theory_db: f64,
}

fn reference_currents(voltages: &[f32], w: &[Vec<f64>], g_max: f64) -> Result<Vec<f64>, String> {
    let rows = voltages.len();
    if rows < 1 || w.len() != rows || w[0].is_empty() {
        return Err("reference_currents: dimension mismatch".into());
    }
    let cols = w[0].len();
    for row in w {
        if row.len() != cols {
            return Err("reference_currents: ragged weight matrix".into());
        }
    }
    let mut i_out = vec![0.0; cols];
    for j in 0..cols {
        let mut sum = 0.0;
        for i in 0..rows {
            sum += voltages[i] as f64 * w[i][j] * g_max;
        }
        i_out[j] = sum;
    }
    Ok(i_out)
}

fn apply_crossbar_physics(
    crossbar: &mut CrossbarArray,
    thermal: &mut ThermalNoiseInjector,
    disturb: &ReadDisturbSimulator,
    opt: &TwoLayerOptions,
    active_row: i32,
) {
    if opt.write_endurance_cycles > 0 {
        let wend = WriteEnduranceSimulator::new(crossbar.config().clone());
        wend.apply_write_cycles(crossbar, opt.write_endurance_cycles);
    }
    if opt.use_persistent_noise {
        thermal.inject_persistent(crossbar);
    }
    for _ in 0..opt.disturb_cycles {
        disturb.apply_disturb(crossbar, active_row, crossbar.config().v_max);
    }
}

fn apply_voltage_with_noise(
    crossbar: &CrossbarArray,
    voltages: &[f32],
    thermal: &mut ThermalNoiseInjector,
    transient: bool,
) -> Result<Vec<f32>, String> {
    let rows = crossbar.rows();
    let cols = crossbar.cols();
    if !transient {
        return crossbar.apply_voltage(voltages);
    }
    let mut gp = vec![vec![0.0f32; cols as usize]; rows as usize];
    let mut gn = vec![vec![0.0f32; cols as usize]; rows as usize];
    for i in 0..rows {
        for j in 0..cols {
            gp[i as usize][j as usize] = crossbar.g_pos_at(i, j);
            gn[i as usize][j as usize] = crossbar.g_neg_at(i, j);
        }
    }
    let mut flat = Vec::with_capacity((2 * rows * cols) as usize);
    for i in 0..rows as usize {
        for j in 0..cols as usize {
            flat.push(gp[i][j]);
        }
    }
    for i in 0..rows as usize {
        for j in 0..cols as usize {
            flat.push(gn[i][j]);
        }
    }
    thermal.inject_transient(&mut flat);
    let mut k = 0;
    for i in 0..rows as usize {
        for j in 0..cols as usize {
            gp[i][j] = flat[k];
            k += 1;
        }
    }
    for i in 0..rows as usize {
        for j in 0..cols as usize {
            gn[i][j] = flat[k];
            k += 1;
        }
    }
    crossbar.apply_voltage_with_g(voltages, &gp, &gn)
}

fn interlayer_process_f32(i: f32, opt: &TwoLayerOptions, cfg: &Config) -> f32 {
    let mut out = circuit_transfer_f32(i, opt.interlayer_circuit, cfg);
    if opt.interlayer_activation != Activation::Identity {
        out = apply_activation_f32(out, opt.interlayer_activation, cfg);
    }
    out
}

fn interlayer_process_f64(i: f64, opt: &TwoLayerOptions, cfg: &Config) -> f64 {
    let mut out = circuit_transfer_f64(i, opt.interlayer_circuit, cfg);
    if opt.interlayer_activation != Activation::Identity {
        out = apply_activation_f64(out, opt.interlayer_activation, cfg);
    }
    out
}

pub fn run_two_layer(
    name: &str,
    cfg: &Config,
    w1: &[Vec<f64>],
    w2: &[Vec<f64>],
    digital_inputs: &[f32],
    opt: &TwoLayerOptions,
) -> Result<TwoLayerResult, String> {
    let in_dim = w1.len();
    if in_dim < 1 || w1[0].is_empty() {
        return Err("run_two_layer: W1 must be non-empty".into());
    }
    let hidden_dim = w1[0].len();
    for row in w1 {
        if row.len() != hidden_dim {
            return Err("run_two_layer: W1 is ragged".into());
        }
    }
    if w2.len() != hidden_dim || w2[0].is_empty() {
        return Err("run_two_layer: W2 row count must match W1 cols".into());
    }
    let out_dim = w2[0].len();
    for row in w2 {
        if row.len() != out_dim {
            return Err("run_two_layer: W2 is ragged".into());
        }
    }
    if digital_inputs.len() != in_dim {
        return Err("run_two_layer: input length must equal W1 rows".into());
    }

    let dac = SimulatedDac::new(cfg.clone());
    let mut crossbar1 = CrossbarArray::new(in_dim as i32, hidden_dim as i32, cfg.clone());
    let mut crossbar2 = CrossbarArray::new(hidden_dim as i32, out_dim as i32, cfg.clone());
    let adc = SimulatedAdc::new(cfg.clone());
    let mut thermal = ThermalNoiseInjector::new(cfg.clone());
    let disturb = ReadDisturbSimulator::new(cfg.clone());

    let wf1: Vec<Vec<f32>> = w1
        .iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();
    let wf2: Vec<Vec<f32>> = w2
        .iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();
    crossbar1.load_weights(&wf1)?;
    crossbar2.load_weights(&wf2)?;

    let active_row1 = if in_dim <= 1 { 0 } else { 2.min(in_dim - 1) };
    let active_row2 = if hidden_dim <= 1 {
        0
    } else {
        2.min(hidden_dim - 1)
    };
    apply_crossbar_physics(
        &mut crossbar1,
        &mut thermal,
        &disturb,
        opt,
        active_row1 as i32,
    );
    apply_crossbar_physics(
        &mut crossbar2,
        &mut thermal,
        &disturb,
        opt,
        active_row2 as i32,
    );

    let voltages1 = dac.convert(digital_inputs);
    let currents1 =
        apply_voltage_with_noise(&crossbar1, &voltages1, &mut thermal, opt.use_transient_noise)?;

    let hidden_processed: Vec<f32> = currents1
        .iter()
        .map(|&c| interlayer_process_f32(c, opt, cfg))
        .collect();

    let dac_inputs_l2: Vec<f32> = hidden_processed
        .iter()
        .map(|&c| {
            let lv = adc.quantize(c);
            adc.level_to_dac_normalized(lv)
        })
        .collect();

    let voltages2 = dac.convert(&dac_inputs_l2);
    let currents2 =
        apply_voltage_with_noise(&crossbar2, &voltages2, &mut thermal, opt.use_transient_noise)?;

    let i1_ref = reference_currents(
        &voltages1,
        w1,
        crossbar1.effective_g_max() as f64,
    )?;

    let hidden_ref_q: Vec<f32> = i1_ref
        .iter()
        .map(|&c| {
            let processed = interlayer_process_f64(c, opt, cfg);
            let lv = adc.quantize(processed as f32);
            adc.level_to_dac_normalized(lv)
        })
        .collect();

    let voltages2_ref = dac.convert(&hidden_ref_q);
    let i2_ref = reference_currents(
        &voltages2_ref,
        w2,
        crossbar2.effective_g_max() as f64,
    )?;

    let mut mse = 0.0;
    let mut max_abs = 0.0f64;
    let mut mean_ref_sq = 0.0;
    for j in 0..out_dim {
        let level = adc.quantize(currents2[j]);
        let recon = adc.reconstruct(level);
        let ref_v = i2_ref[j];
        let err = recon as f64 - ref_v;
        mse += err * err;
        max_abs = max_abs.max(err.abs());
        mean_ref_sq += ref_v * ref_v;
    }
    mse /= out_dim as f64;
    mean_ref_sq /= out_dim as f64;

    let snr_db = if mse > 1e-30 {
        10.0 * (mean_ref_sq / mse).log10()
    } else {
        200.0
    };

    let ps = cfg.i_range as f64 * cfg.i_range as f64 / 8.0;
    let delta = cfg.i_range as f64 / adc.max_level() as f64;
    let pq = delta * delta / 12.0;
    let snr_adc_theory_db = 10.0 * (ps / pq).log10();

    Ok(TwoLayerResult {
        name: name.to_string(),
        n_bits: cfg.n_bits_adc,
        noise_stddev: cfg.noise_stddev,
        disturb_cycles: opt.disturb_cycles,
        endurance_cycles: opt.write_endurance_cycles,
        mse,
        max_abs_err: max_abs,
        snr_db,
        snr_adc_theory_db,
    })
}
