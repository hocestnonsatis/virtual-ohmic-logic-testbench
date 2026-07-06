use volt_core::{
    run_two_layer, Config, CrossbarArray, ReadDisturbSimulator, SimulatedAdc, SimulatedDac,
    ThermalNoiseInjector, TwoLayerOptions,
};

fn scenario_a_mse() -> f64 {
    let w = vec![
        vec![0.8, -0.3, 0.5, -0.1],
        vec![-0.6, 0.9, -0.2, 0.7],
        vec![0.1, -0.8, 0.4, -0.5],
        vec![0.3, 0.2, -0.9, 0.6],
    ];
    let inputs = vec![0.9f32, 0.4, 0.7, 0.2];

    let cfg = Config::default();
    let dac = SimulatedDac::new(cfg.clone());
    let mut crossbar = CrossbarArray::new(4, 4, cfg.clone());
    let adc = SimulatedAdc::new(cfg.clone());

    let wf: Vec<Vec<f32>> = w
        .iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();
    crossbar.load_weights(&wf).unwrap();

    let voltages = dac.convert(&inputs);

    let mut i_ref = vec![0.0; 4];
    for j in 0..4 {
        let mut sum = 0.0;
        for i in 0..4 {
            sum += voltages[i] as f64 * w[i][j] * cfg.g_max as f64;
        }
        i_ref[j] = sum;
    }

    let currents = crossbar.apply_voltage(&voltages).unwrap();

    let mut mse = 0.0;
    for j in 0..4 {
        let level = adc.quantize(currents[j]);
        assert!(level <= adc.max_level());
        let recon = adc.reconstruct(level);
        let err = recon as f64 - i_ref[j];
        mse += err * err;
    }
    mse / 4.0
}

fn assert_conductances_ok(cb: &CrossbarArray) {
    let gmax = cb.config().g_max;
    for i in 0..cb.rows() {
        for j in 0..cb.cols() {
            let gp = cb.g_pos_at(i, j);
            let gn = cb.g_neg_at(i, j);
            assert!(gp >= 0.0 && gp <= gmax && gn >= 0.0 && gn <= gmax);
        }
    }
}

#[test]
fn equivalence_regression() {
    let mse = scenario_a_mse();
    assert!(
        mse < 1e-6,
        "Scenario A MSE must be < 1e-6, got {mse}"
    );

    let cfg = Config::default();
    let mut cb = CrossbarArray::new(4, 4, cfg.clone());
    let w = vec![
        vec![0.5f32, -0.2, 0.0, 0.3],
        vec![-0.1, 0.4, 0.6, -0.9],
        vec![0.2, 0.2, -0.5, 0.1],
        vec![0.0, 0.8, -0.3, 0.4],
    ];
    cb.load_weights(&w).unwrap();
    assert_conductances_ok(&cb);

    let mut th = ThermalNoiseInjector::new(cfg.clone());
    th.inject_persistent(&mut cb);
    assert_conductances_ok(&cb);

    let rd = ReadDisturbSimulator::new(cfg.clone());
    rd.apply_disturb(&mut cb, 1, cfg.v_max);
    assert_conductances_ok(&cb);

    let adc = SimulatedAdc::new(cfg.clone());
    let mut c = cfg.i_min - cfg.i_range * 0.1;
    while c <= cfg.i_min + cfg.i_range * 1.1 {
        let lv = adc.quantize(c);
        assert!(lv >= 0 && lv <= adc.max_level());
        c += cfg.i_range * 0.05;
    }

    let w1 = vec![
        vec![0.8, -0.3, 0.5, -0.1],
        vec![-0.6, 0.9, -0.2, 0.7],
        vec![0.1, -0.8, 0.4, -0.5],
        vec![0.3, 0.2, -0.9, 0.6],
    ];
    let mut w2 = vec![vec![0.0; 4]; 4];
    for i in 0..4 {
        w2[i][i] = 0.5;
    }
    let inputs = vec![0.9f32, 0.4, 0.7, 0.2];
    let f_res = run_two_layer(
        "F_multilayer",
        &cfg,
        &w1,
        &w2,
        &inputs,
        &TwoLayerOptions::default(),
    )
    .unwrap();
    assert!(f_res.mse.is_finite() && f_res.mse >= 0.0);
}
