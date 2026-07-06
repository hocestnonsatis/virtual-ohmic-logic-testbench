use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::crossbar::CrossbarArray;
use crate::dac::SimulatedDac;

fn deterministic_weight(i: i32, j: i32) -> f32 {
    let ui = (i as u32).wrapping_mul(2654435761);
    let uj = (j as u32).wrapping_mul(2246822519);
    let u = (ui ^ uj ^ (ui << 3)) % 2001;
    let w = u as i32 as f32 / 1000.0 - 1.0;
    w.clamp(-1.0, 1.0)
}

pub fn run_benchmark_suite(cfg: &Config) {
    let dac = SimulatedDac::new(cfg.clone());
    let sizes = [4, 8, 16, 32, 64, 128];
    let target = Duration::from_millis(40);

    println!("[benchmark] n×n crossbar MAC (Ohm/KCL); timing steady-state forwards");

    let mut csv = File::create("benchmark.csv").expect("create benchmark.csv");
    writeln!(
        csv,
        "n,repetitions,total_s,forwards_per_s,gmac_per_s,ns_per_forward"
    )
    .unwrap();

    for &n in &sizes {
        let mut cb = CrossbarArray::new(n, n, cfg.clone());
        let mut wf = vec![vec![0.0f32; n as usize]; n as usize];
        for i in 0..n {
            for j in 0..n {
                wf[i as usize][j as usize] = deterministic_weight(i, j);
            }
        }
        cb.load_weights(&wf).unwrap();

        let mut inputs = vec![0.5f32; n as usize];
        for i in 0..n {
            inputs[i as usize] =
                0.15 + 0.7 * i as f32 / (n - 1).max(1) as f32;
        }
        let voltages = dac.convert(&inputs);

        let mut sink = 0.0f32;
        for _ in 0..8 {
            let cur = cb.apply_voltage(&voltages).unwrap();
            for &x in &cur {
                sink += x;
            }
        }

        let mut reps: i64 = 0;
        let t_begin = Instant::now();
        while t_begin.elapsed() < target {
            for _ in 0..64 {
                let cur = cb.apply_voltage(&voltages).unwrap();
                for &x in &cur {
                    sink += x;
                }
                reps += 1;
            }
        }
        let _ = sink;
        let total_s = t_begin.elapsed().as_secs_f64();
        if total_s <= 0.0 || reps <= 0 {
            continue;
        }
        let fwd_per_s = reps as f64 / total_s;
        let mac_per_fwd = (n * n) as f64;
        let gmac_per_s = mac_per_fwd * fwd_per_s / 1e9;
        let ns_per_fwd = 1e9 / fwd_per_s;

        println!(
            "  n={n}  reps={reps}  {fwd_per_s} fwd/s  {gmac_per_s} GMAC/s"
        );
        writeln!(
            csv,
            "{n},{reps},{total_s:.9},{fwd_per_s:.9},{gmac_per_s:.9},{ns_per_fwd:.9}"
        )
        .unwrap();
    }

    println!("[benchmark] wrote benchmark.csv (cwd)");
}
