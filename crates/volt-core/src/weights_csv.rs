use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub const K_MAX_WEIGHTS_ROWS: i32 = 512;
pub const K_MAX_WEIGHTS_COLS: i32 = 512;

fn trim(s: &str) -> &str {
    s.trim()
}

fn parse_line_row(line: &str) -> Result<Vec<f64>, String> {
    let mut row = Vec::new();
    let mut pos = 0;
    let bytes = line.as_bytes();
    while pos < bytes.len() {
        let comma = line[pos..].find(',').map(|i| pos + i);
        let end = comma.unwrap_or(bytes.len());
        let tok = trim(&line[pos..end]);
        if tok.is_empty() {
            return Err("CSV: empty field in row".into());
        }
        let v: f64 = tok
            .parse()
            .map_err(|_| format!("CSV: invalid number: {tok}"))?;
        row.push(v);
        if comma.is_none() {
            break;
        }
        pos = end + 1;
    }
    if row.is_empty() {
        return Err("CSV: empty row".into());
    }
    Ok(row)
}

pub fn load_weights_csv_file(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let f = File::open(path).map_err(|_| format!("cannot open weights file: {path}"))?;
    let reader = BufReader::new(f);
    let mut out: Vec<Vec<f64>> = Vec::new();
    let mut line_no = 0;
    for line_result in reader.lines() {
        line_no += 1;
        let line = line_result.map_err(|e| format!("read error line {line_no}: {e}"))?;
        let line = trim(&line);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row = parse_line_row(line).map_err(|e| format!("line {line_no}: {e}"))?;
        if !out.is_empty() && row.len() != out[0].len() {
            return Err(format!("CSV: row width mismatch at line {line_no}"));
        }
        out.push(row);
    }
    if out.is_empty() {
        return Err("CSV: no data rows".into());
    }
    let n_rows = out.len();
    let n_cols = out[0].len();
    if n_rows > K_MAX_WEIGHTS_ROWS as usize {
        return Err("CSV: row count exceeds k_max_weights_rows".into());
    }
    if n_cols > K_MAX_WEIGHTS_COLS as usize {
        return Err("CSV: column count exceeds k_max_weights_cols".into());
    }
    for (i, row) in out.iter_mut().enumerate() {
        for (j, w) in row.iter_mut().enumerate() {
            if *w < -1.0 || *w > 1.0 {
                eprintln!(
                    "[weights_csv] warning: entry ({i},{j}) = {w} outside [-1,1]; clamping"
                );
                *w = w.clamp(-1.0, 1.0);
            }
        }
    }
    Ok(out)
}

pub fn write_weights_csv_file(path: &str, weights: &[Vec<f64>]) -> Result<(), String> {
    if weights.is_empty() {
        return Err("write_weights_csv: empty matrix".into());
    }
    let cols = weights[0].len();
    if cols == 0 {
        return Err("write_weights_csv: empty row".into());
    }
    for (i, row) in weights.iter().enumerate() {
        if row.len() != cols {
            return Err(format!("write_weights_csv: ragged row {i}"));
        }
    }
    if weights.len() > K_MAX_WEIGHTS_ROWS as usize {
        return Err("write_weights_csv: row count exceeds k_max_weights_rows".into());
    }
    if cols > K_MAX_WEIGHTS_COLS as usize {
        return Err("write_weights_csv: column count exceeds k_max_weights_cols".into());
    }
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("write_weights_csv: cannot create {path}: {e}"))?;
    writeln!(f, "# VOLT weights export")
        .map_err(|e| format!("write_weights_csv: write error: {e}"))?;
    for row in weights {
        let line: Vec<String> = row.iter().map(|v| format!("{v:.17}")).collect();
        writeln!(f, "{}", line.join(","))
            .map_err(|e| format!("write_weights_csv: write error: {e}"))?;
    }
    Ok(())
}

pub fn load_inputs_csv_file(path: &str, expected_n: i32) -> Result<Vec<f32>, String> {
    if expected_n < 1 {
        return Err("inputs: expected_n must be positive".into());
    }
    let f = File::open(path).map_err(|_| format!("cannot open inputs file: {path}"))?;
    let reader = BufReader::new(f);
    let mut acc: Vec<f64> = Vec::new();
    let mut line_no = 0;
    for line_result in reader.lines() {
        line_no += 1;
        let line = line_result.map_err(|e| format!("read error line {line_no}: {e}"))?;
        let line = trim(&line);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row = parse_line_row(line).map_err(|e| format!("line {line_no}: {e}"))?;
        acc.extend(row);
    }
    if acc.len() != expected_n as usize {
        return Err(format!(
            "inputs: expected {expected_n} values, got {}",
            acc.len()
        ));
    }
    let mut out = Vec::with_capacity(expected_n as usize);
    for (i, &x) in acc.iter().enumerate() {
        let mut v = x as f32;
        if v < 0.0 || v > 1.0 {
            eprintln!(
                "[inputs_csv] warning: input {i} = {v} outside [0,1]; clamping"
            );
            v = v.clamp(0.0, 1.0);
        }
        out.push(v);
    }
    Ok(out)
}
