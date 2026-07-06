use std::fs;
use std::io::Write;

use volt_core::{
    apply_activation_f32, circuit_transfer_f32, load_config_from_json, load_inputs_csv_file,
    load_weights_csv_file, parse_circuit_model, parse_iv_model, run_two_layer, Activation,
    CircuitModel, Config, CrossbarArray, IvModel, ReadDisturbSimulator, SimulatedAdc,
    SimulatedDac, ThermalNoiseInjector, TwoLayerOptions, WriteEnduranceSimulator, cell_current,
};

fn assert_near(a: f32, b: f32, eps: f32, msg: &str) {
    assert!(
        (a - b).abs() <= eps,
        "FAIL: {msg} ({a} vs {b})"
    );
}

#[test]
fn dac_tests() {
    let cfg = Config::default();
    let dac = SimulatedDac::new(cfg.clone());
    assert_near(dac.convert(&[0.0])[0], cfg.v_min, 1e-6, "DAC 0 -> V_min");
    assert_near(dac.convert(&[1.0])[0], cfg.v_max, 1e-5, "DAC 1 -> V_max");
    let mid = (cfg.v_min + cfg.v_max) * 0.5;
    assert_near(dac.convert(&[0.5])[0], mid, 1e-5, "DAC 0.5 mid");

    let bad = dac.convert(&[1.5]);
    assert_near(bad[0], cfg.v_max, 1e-5, "DAC 1.5 clamped");

    let vr = dac.convert_u8(&[0, 255, 128]);
    assert_near(vr[0], cfg.v_min, 1e-5, "DAC uint8 0");
    assert_near(vr[1], cfg.v_max, 1e-5, "DAC uint8 255");
    let exp128 = cfg.v_min + (128.0 / 255.0) * (cfg.v_max - cfg.v_min);
    assert_near(vr[2], exp128, 1e-5, "DAC uint8 128");
}

#[test]
fn adc_tests() {
    let mut cfg = Config::default();
    cfg.n_bits_adc = 8;
    let adc8 = SimulatedAdc::new(cfg.clone());
    let mid = cfg.i_min + cfg.i_range * 0.5;
    let l8 = adc8.quantize(mid);
    assert!((l8 - 127).abs() <= 1, "8-bit half range");

    cfg.n_bits_adc = 4;
    let adc4 = SimulatedAdc::new(cfg.clone());
    let l4 = adc4.quantize(mid);
    assert!((l4 - 7).abs() <= 1, "4-bit half range");

    cfg.n_bits_adc = 8;
    let adc = SimulatedAdc::new(cfg.clone());
    assert_eq!(adc.quantize(cfg.i_min), 0);
    assert_eq!(adc.quantize(cfg.i_min + cfg.i_range), adc.max_level());
    assert_eq!(
        adc.quantize(cfg.i_min + cfg.i_range * 100.0),
        adc.max_level()
    );
    assert_near(adc.level_to_dac_normalized(0), 0.0, 1e-7, "L0");
    assert_near(
        adc.level_to_dac_normalized(adc.max_level()),
        1.0,
        1e-6,
        "Lmax",
    );
}

#[test]
fn activation_tests() {
    let c = Config::default();
    assert_near(
        apply_activation_f32(-1.0e-5, Activation::ReLU, &c),
        0.0,
        1e-12,
        "ReLU neg",
    );
    assert_near(
        apply_activation_f32(3.0e-5, Activation::ReLU, &c),
        3.0e-5,
        1e-12,
        "ReLU pos",
    );
    let ys = apply_activation_f32(0.0, Activation::Sigmoid, &c);
    assert!(ys >= c.i_min && ys <= c.i_min + c.i_range);
}

#[test]
fn crossbar_tests() {
    let cfg = Config::default();
    let mut xb = CrossbarArray::new(2, 2, cfg.clone());
    xb.load_weights(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    assert_near(xb.g_pos_at(0, 0), cfg.g_max, 1e-5 * cfg.g_max, "w=1 G_pos");
    assert_near(xb.g_neg_at(0, 0), 0.0, 1e-5 * cfg.g_max, "w=1 G_neg");

    let mut xbn = CrossbarArray::new(2, 2, cfg.clone());
    xbn.load_weights(&[vec![-1.0, 0.0], vec![0.0, -1.0]]).unwrap();
    assert_near(xbn.g_pos_at(0, 0), 0.0, 1e-5 * cfg.g_max, "w=-1 G_pos");

    let mut xr = CrossbarArray::new(2, 2, cfg.clone());
    let wr = vec![vec![0.3, -0.7], vec![0.1, 0.9]];
    xr.load_weights(&wr).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_near(
                xr.get_effective_weight(i, j),
                wr[i as usize][j as usize],
                1e-4,
                "effective weight",
            );
        }
    }

    let i = xb.apply_voltage(&[0.3, 0.7]).unwrap();
    assert_near(i[0], 0.3 * cfg.g_max, 1e-4 * cfg.g_max, "identity I0");
    assert_near(i[1], 0.7 * cfg.g_max, 1e-4 * cfg.g_max, "identity I1");
}

#[test]
fn noise_tests() {
    let mut c0 = Config::default();
    c0.noise_stddev = 0.0;
    let mut n0 = ThermalNoiseInjector::new(c0);
    let g = vec![1.0, 2.0, 3.0];
    let mut gcopy = g.clone();
    n0.inject_transient(&mut gcopy);
    assert_near(gcopy[0], g[0], 1e-7, "zero stddev");

    let mut c1 = Config::default();
    c1.noise_stddev = 0.01;
    c1.noise_seed = 12345;
    let mut a = ThermalNoiseInjector::new(c1.clone());
    let mut b = ThermalNoiseInjector::new(c1.clone());
    let mut v1 = vec![c1.g_max * 0.5, c1.g_max * 0.25];
    let mut v2 = v1.clone();
    a.inject_transient(&mut v1);
    b.inject_transient(&mut v2);
    assert_near(v1[0], v2[0], 1e-6, "seed match 0");
    assert_near(v1[1], v2[1], 1e-6, "seed match 1");

    let mut cd = Config::default();
    cd.disturb_alpha = 1e-5;
    cd.disturb_ratio = 0.03;
    let mut arr = CrossbarArray::new(4, 4, cd.clone());
    arr.load_weights(&vec![vec![0.0; 4]; 4]).unwrap();
    let g0 = arr.g_pos_at(1, 0);
    let ds = ReadDisturbSimulator::new(cd.clone());
    let vapp = 0.01f32;
    for _ in 0..1000 {
        ds.apply_disturb(&mut arr, 2, vapp);
    }
    let expected = 1000.0 * cd.disturb_alpha * (vapp * cd.disturb_ratio);
    assert_near(arr.g_pos_at(1, 0) - g0, expected, 1e-7, "read disturb");

    let mut ce = Config::default();
    ce.write_endurance_lambda = 1e-5;
    let mut ar = CrossbarArray::new(2, 2, ce.clone());
    ar.load_weights(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    let g0 = ar.effective_g_max();
    WriteEnduranceSimulator::new(ce.clone()).apply_write_cycles(&mut ar, 100_000);
    assert!(ar.effective_g_max() < g0);
    let scale = (-1e-5f32 * 100_000.0).exp();
    assert_near(ar.effective_g_max(), ce.g_max * scale, 1e-5 * ce.g_max, "endurance");
}

#[test]
fn json_config_tests() {
    let mut c = Config::default();
    load_config_from_json(r#"{"G_max":2e-4,"noise_seed":99,"n_bits_adc":6}"#, &mut c).unwrap();
    assert_near(c.g_max, 2e-4, 1e-12, "G_max");
    assert_eq!(c.noise_seed, 99);
    assert_eq!(c.n_bits_adc, 6);

    let mut c2 = Config::default();
    load_config_from_json("{}", &mut c2).unwrap();
    assert_near(c2.g_max, Config::default().g_max, 1e-12, "empty JSON");

    let mut c3 = Config::default();
    load_config_from_json(
        r#"{"iv_model":"soft_saturation","interlayer_circuit":"diode_rectifier"}"#,
        &mut c3,
    )
    .unwrap();
    assert_eq!(c3.iv_model, IvModel::SoftSaturation);
    assert_eq!(c3.interlayer_circuit, CircuitModel::DiodeRectifier);
}

#[test]
fn csv_tests() {
    let w_path = "volt_test_weights.csv";
    {
        let mut f = fs::File::create(w_path).unwrap();
        writeln!(f, "# identity").unwrap();
        writeln!(f, "1,0,0,0").unwrap();
        writeln!(f, "0,1,0,0").unwrap();
        writeln!(f, "0,0,1,0").unwrap();
        writeln!(f, "0,0,0,1").unwrap();
    }
    let w = load_weights_csv_file(w_path).unwrap();
    let _ = fs::remove_file(w_path);
    assert_near(w[0][0] as f32, 1.0, 1e-9, "CSV W00");

    let in_path = "volt_test_inputs.csv";
    {
        let mut f = fs::File::create(in_path).unwrap();
        writeln!(f, "0.2").unwrap();
        writeln!(f, "0.5").unwrap();
        writeln!(f, "# c").unwrap();
        writeln!(f, "0.8").unwrap();
    }
    let inputs = load_inputs_csv_file(in_path, 3).unwrap();
    let _ = fs::remove_file(in_path);
    assert_near(inputs[0], 0.2, 1e-9, "in0");
    assert_near(inputs[2], 0.8, 1e-9, "in2");
}

#[test]
fn iv_and_circuit_tests() {
    let mut c = Config::default();
    let v = 0.5f32;
    let g = 1e-4f32;
    assert_near(cell_current(v, g, IvModel::Linear, &c), g * v, 1e-12, "linear");
    c.iv_exponent = 2.0;
    assert_near(
        cell_current(v, g, IvModel::PowerLaw, &c),
        g * v * v,
        1e-12,
        "power law",
    );
    c.iv_v_sat = 1.0;
    assert_near(
        cell_current(v, g, IvModel::SoftSaturation, &c),
        g * v / (1.0 + v),
        1e-12,
        "soft sat",
    );

    assert!(parse_iv_model("power_law").is_some());
    assert!(parse_circuit_model("diode_rectifier").is_some());
    let mut cfg = Config::default();
    cfg.circuit_i_threshold = 1e-6;
    assert_near(
        circuit_transfer_f32(5e-6, CircuitModel::DiodeRectifier, &cfg),
        4e-6,
        1e-12,
        "diode",
    );
}

#[test]
fn two_layer_mxk() {
    let w1 = vec![vec![0.5, -0.2], vec![0.1, 0.3]];
    let w2 = vec![vec![0.4, -0.1, 0.2], vec![-0.3, 0.6, 0.0]];
    let inputs = vec![0.5f32, 0.7f32];
    let r = run_two_layer(
        "test_mxk",
        &Config::default(),
        &w1,
        &w2,
        &inputs,
        &TwoLayerOptions::default(),
    )
    .unwrap();
    assert!(r.mse >= 0.0);
    assert!(r.snr_db.is_finite());
}
