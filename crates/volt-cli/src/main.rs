use std::fs::File;
use std::io::Write;

use clap::Parser;
use volt_core::{
    apply_activation_f32, apply_activation_f64, circuit_model_name, iv_model_name,
    load_config_from_json_file, load_inputs_csv_file, load_weights_csv_file, run_benchmark_suite,
    run_two_layer, Activation, CircuitModel, Config, CrossbarArray, IvModel, ReadDisturbSimulator,
    SimulatedAdc, SimulatedDac, ThermalNoiseInjector, TwoLayerOptions, TwoLayerResult,
    WriteEnduranceSimulator,
};

#[derive(Parser)]
#[command(name = "volt")]
#[command(about = "VOLT — Virtual Ohmic Logic Testbench")]
struct Cli {
    #[arg(long)]
    config: Option<String>,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long)]
    weights2: Option<String>,
    #[arg(long)]
    inputs: Option<String>,
    #[arg(long)]
    benchmark: bool,
}

struct ScenarioResult {
    name: String,
    n_bits: i32,
    noise_stddev: f32,
    disturb_cycles: i32,
    endurance_cycles: i32,
    mse: f64,
    max_abs_err: f64,
    snr_db: f64,
    snr_adc_theory_db: f64,
    iv_model: String,
    interlayer_circuit: String,
}

fn reference_currents(
    voltages: &[f32],
    w: &[Vec<f64>],
    g_max: f64,
) -> Result<Vec<f64>, String> {
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

fn run_scenario(
    name: &str,
    cfg: Config,
    disturb_cycles: i32,
    w_double: &[Vec<f64>],
    digital_inputs: &[f32],
    use_transient_noise: bool,
    use_persistent_noise: bool,
    print_currents: bool,
    activation: Activation,
    write_endurance_cycles: i32,
) -> Result<ScenarioResult, String> {
    let rows = w_double.len();
    if rows < 1 || w_double[0].is_empty() {
        return Err("run_scenario: empty weight matrix".into());
    }
    let cols = w_double[0].len();
    for row in w_double {
        if row.len() != cols {
            return Err("run_scenario: ragged weight matrix".into());
        }
    }
    if digital_inputs.len() != rows {
        return Err("run_scenario: input length must equal weight rows".into());
    }

    let dac = SimulatedDac::new(cfg.clone());
    let mut crossbar = CrossbarArray::new(rows as i32, cols as i32, cfg.clone());
    let adc = SimulatedAdc::new(cfg.clone());
    let mut thermal = ThermalNoiseInjector::new(cfg.clone());
    let disturb = ReadDisturbSimulator::new(cfg.clone());

    let wf: Vec<Vec<f32>> = w_double
        .iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();
    crossbar.load_weights(&wf)?;

    if write_endurance_cycles > 0 {
        let wend = WriteEnduranceSimulator::new(cfg.clone());
        wend.apply_write_cycles(&mut crossbar, write_endurance_cycles);
    }
    if use_persistent_noise {
        thermal.inject_persistent(&mut crossbar);
    }

    let active_row = if rows <= 1 { 0 } else { 2.min(rows - 1) };
    for _ in 0..disturb_cycles {
        disturb.apply_disturb(&mut crossbar, active_row as i32, cfg.v_max);
    }

    let voltages = dac.convert(digital_inputs);

    let mut gp = vec![vec![0.0f32; cols]; rows];
    let mut gn = vec![vec![0.0f32; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            gp[i][j] = crossbar.g_pos_at(i as i32, j as i32);
            gn[i][j] = crossbar.g_neg_at(i as i32, j as i32);
        }
    }

    let mut currents = if use_transient_noise {
        let mut flat = Vec::with_capacity(2 * rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                flat.push(gp[i][j]);
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                flat.push(gn[i][j]);
            }
        }
        thermal.inject_transient(&mut flat);
        let mut k = 0;
        for i in 0..rows {
            for j in 0..cols {
                gp[i][j] = flat[k];
                k += 1;
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                gn[i][j] = flat[k];
                k += 1;
            }
        }
        crossbar.apply_voltage_with_g(&voltages, &gp, &gn)?
    } else {
        crossbar.apply_voltage(&voltages)?
    };

    if activation != Activation::Identity {
        for c in currents.iter_mut() {
            *c = apply_activation_f32(*c, activation, &cfg);
        }
    }

    if print_currents {
        print!("[Scenario A] raw I_net before ADC (A): ");
        for (j, &c) in currents.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{c:.6e}");
        }
        println!();
    }

    let mut i_ref = reference_currents(
        &voltages,
        w_double,
        crossbar.effective_g_max() as f64,
    )?;
    if activation != Activation::Identity {
        for r in i_ref.iter_mut() {
            *r = apply_activation_f64(*r, activation, &cfg);
        }
    }

    let mut mse = 0.0;
    let mut max_abs = 0.0f64;
    let mut mean_ref_sq = 0.0;
    for j in 0..cols {
        let level = adc.quantize(currents[j]);
        let recon = adc.reconstruct(level);
        let ref_v = i_ref[j];
        let err = recon as f64 - ref_v;
        mse += err * err;
        max_abs = max_abs.max(err.abs());
        mean_ref_sq += ref_v * ref_v;
    }
    mse /= cols as f64;
    mean_ref_sq /= cols as f64;

    let snr_db = if mse > 1e-30 {
        10.0 * (mean_ref_sq / mse).log10()
    } else {
        200.0
    };

    let ps = cfg.i_range as f64 * cfg.i_range as f64 / 8.0;
    let delta = cfg.i_range as f64 / adc.max_level() as f64;
    let pq = delta * delta / 12.0;
    let snr_adc_theory_db = 10.0 * (ps / pq).log10();

    Ok(ScenarioResult {
        name: name.to_string(),
        n_bits: cfg.n_bits_adc,
        noise_stddev: cfg.noise_stddev,
        disturb_cycles,
        endurance_cycles: write_endurance_cycles,
        mse,
        max_abs_err: max_abs,
        snr_db,
        snr_adc_theory_db,
        iv_model: iv_model_name(cfg.iv_model).to_string(),
        interlayer_circuit: circuit_model_name(cfg.interlayer_circuit).to_string(),
    })
}

fn from_two_layer(t: &TwoLayerResult, cfg: &Config) -> ScenarioResult {
    ScenarioResult {
        name: t.name.clone(),
        n_bits: t.n_bits,
        noise_stddev: t.noise_stddev,
        disturb_cycles: t.disturb_cycles,
        endurance_cycles: t.endurance_cycles,
        mse: t.mse,
        max_abs_err: t.max_abs_err,
        snr_db: t.snr_db,
        snr_adc_theory_db: t.snr_adc_theory_db,
        iv_model: iv_model_name(cfg.iv_model).to_string(),
        interlayer_circuit: circuit_model_name(cfg.interlayer_circuit).to_string(),
    }
}

const DEFAULT_W: [[f64; 4]; 4] = [
    [0.8, -0.3, 0.5, -0.1],
    [-0.6, 0.9, -0.2, 0.7],
    [0.1, -0.8, 0.4, -0.5],
    [0.3, 0.2, -0.9, 0.6],
];
const DEFAULT_INPUTS: [f32; 4] = [0.9, 0.4, 0.7, 0.2];

fn default_w() -> Vec<Vec<f64>> {
    DEFAULT_W.iter().map(|row| row.to_vec()).collect()
}

fn default_inputs_for_n(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![0.5; n];
    }
    (0..n)
        .map(|i| 0.15 + 0.7 * i as f32 / (n - 1) as f32)
        .collect()
}

fn default_w2_diagonal(n: usize, diag: f64) -> Vec<Vec<f64>> {
    let mut w2 = vec![vec![0.0; n]; n];
    for i in 0..n {
        w2[i][i] = diag;
    }
    w2
}

fn main() {
    let cli = Cli::parse();
    let mut defaults = Config::default();

    if let Some(ref path) = cli.config {
        if let Err(err) = load_config_from_json_file(path, &mut defaults) {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }

    if cli.benchmark {
        run_benchmark_suite(&defaults);
        return;
    }

    let mut w = default_w();
    if let Some(ref path) = cli.weights {
        match load_weights_csv_file(path) {
            Ok(loaded) => w = loaded,
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
    }
    let n_rows = w.len();
    let n_cols = w[0].len();

    let mut w2 = default_w2_diagonal(n_cols, 0.5);
    if let Some(ref path) = cli.weights2 {
        match load_weights_csv_file(path) {
            Ok(loaded) => w2 = loaded,
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        if w2.len() != n_cols {
            eprintln!(
                "error: --weights2 must have {n_cols} rows (matching column count of --weights)"
            );
            std::process::exit(1);
        }
    }

    let inputs = if let Some(ref path) = cli.inputs {
        match load_inputs_csv_file(path, n_rows as i32) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
    } else if cli.weights.is_none() {
        DEFAULT_INPUTS.to_vec()
    } else {
        default_inputs_for_n(n_rows)
    };

    let mut results: Vec<ScenarioResult> = Vec::new();

    macro_rules! push_scenario {
        ($expr:expr) => {
            match $expr {
                Ok(r) => results.push(r),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        };
    }

    {
        let cfg = defaults.clone();
        push_scenario!(run_scenario(
            "A_ideal",
            cfg,
            0,
            &w,
            &inputs,
            false,
            false,
            true,
            Activation::Identity,
            0
        ));
    }
    {
        let mut cfg = defaults.clone();
        cfg.n_bits_adc = 4;
        push_scenario!(run_scenario(
            "B_low_adc",
            cfg,
            0,
            &w,
            &inputs,
            false,
            false,
            false,
            Activation::Identity,
            0
        ));
    }
    {
        let mut cfg = defaults.clone();
        cfg.noise_stddev = 0.005 * cfg.g_max;
        push_scenario!(run_scenario(
            "C_thermal",
            cfg,
            0,
            &w,
            &inputs,
            true,
            false,
            false,
            Activation::Identity,
            0
        ));
    }
    {
        let cfg = defaults.clone();
        push_scenario!(run_scenario(
            "D_read_disturb",
            cfg,
            1000,
            &w,
            &inputs,
            false,
            false,
            false,
            Activation::Identity,
            0
        ));
    }
    {
        let mut cfg = defaults.clone();
        cfg.n_bits_adc = 4;
        cfg.noise_stddev = 0.005 * cfg.g_max;
        push_scenario!(run_scenario(
            "E_combined",
            cfg,
            1000,
            &w,
            &inputs,
            true,
            true,
            false,
            Activation::Identity,
            0
        ));
    }
    {
        let cfg = defaults.clone();
        let opt = TwoLayerOptions::default();
        match run_two_layer("F_multilayer", &cfg, &w, &w2, &inputs, &opt) {
            Ok(t) => results.push(from_two_layer(&t, &cfg)),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
    {
        let cfg = defaults.clone();
        let mut opt = TwoLayerOptions::default();
        opt.interlayer_circuit = CircuitModel::DiodeRectifier;
        match run_two_layer("F_multilayer_relu", &cfg, &w, &w2, &inputs, &opt) {
            Ok(t) => results.push(from_two_layer(&t, &cfg)),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
    {
        let cfg = defaults.clone();
        push_scenario!(run_scenario(
            "G_relu",
            cfg,
            0,
            &w,
            &inputs,
            false,
            false,
            false,
            Activation::ReLU,
            0
        ));
    }
    {
        let cfg = defaults.clone();
        push_scenario!(run_scenario(
            "H_sigmoid",
            cfg,
            0,
            &w,
            &inputs,
            false,
            false,
            false,
            Activation::Sigmoid,
            0
        ));
    }
    {
        let mut cfg = defaults.clone();
        cfg.write_endurance_lambda = 1e-5;
        push_scenario!(run_scenario(
            "I_write_endurance",
            cfg,
            0,
            &w,
            &inputs,
            false,
            false,
            false,
            Activation::Identity,
            80000
        ));
    }
    {
        let mut cfg = defaults.clone();
        cfg.iv_model = IvModel::PowerLaw;
        cfg.iv_exponent = 1.5;
        push_scenario!(run_scenario(
            "J_nonlinear_iv",
            cfg,
            0,
            &w,
            &inputs,
            false,
            false,
            false,
            Activation::Identity,
            0
        ));
    }
    {
        let mut cfg = defaults.clone();
        cfg.iv_model = IvModel::SoftSaturation;
        cfg.interlayer_circuit = CircuitModel::TunableSigmoid;
        let mut opt = TwoLayerOptions::default();
        opt.interlayer_circuit = CircuitModel::TunableSigmoid;
        match run_two_layer("K_interlayer_circuit", &cfg, &w, &w2, &inputs, &opt) {
            Ok(t) => results.push(from_two_layer(&t, &cfg)),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }

    for r in &results {
        println!("=== {} ===", r.name);
        println!("  MSE: {:.6e}", r.mse);
        println!("  Max absolute error (A): {:.8}", r.max_abs_err);
        println!(
            "  SNR (dB, measured mean(I_ref^2)/MSE): {:.8}",
            r.snr_db
        );
        println!(
            "  SQNR (dB, ADC full-scale sine vs Delta^2/12): {:.8}",
            r.snr_adc_theory_db
        );
        if r.endurance_cycles > 0 {
            println!(
                "  Write endurance cycles (modeled): {}",
                r.endurance_cycles
            );
        }
        println!();
    }

    let mut csv = File::create("results.csv").expect("create results.csv");
    writeln!(
        csv,
        "scenario,n_bits,noise_stddev,disturb_cycles,endurance_cycles,iv_model,interlayer_circuit,mse,max_abs_err,snr_measured_db,snr_adc_theory_db"
    )
    .unwrap();
    for r in &results {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12}",
            r.name,
            r.n_bits,
            r.noise_stddev,
            r.disturb_cycles,
            r.endurance_cycles,
            r.iv_model,
            r.interlayer_circuit,
            r.mse,
            r.max_abs_err,
            r.snr_db,
            r.snr_adc_theory_db
        )
        .unwrap();
    }

    {
        let cfg = defaults.clone();
        let mut cb = CrossbarArray::new(n_rows as i32, n_cols as i32, cfg.clone());
        let wf: Vec<Vec<f32>> = w
            .iter()
            .map(|row| row.iter().map(|&x| x as f32).collect())
            .collect();
        cb.load_weights(&wf).unwrap();
        let ds = ReadDisturbSimulator::new(cfg.clone());
        let drift_row = if n_rows <= 1 { 0 } else { 2.min(n_rows - 1) };
        for _ in 0..1000 {
            ds.apply_disturb(&mut cb, drift_row as i32, cfg.v_max);
        }
        ds.log_drift_report(&cb);
    }
}
