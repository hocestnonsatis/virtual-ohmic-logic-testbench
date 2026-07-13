use crate::weights_csv::{K_MAX_WEIGHTS_COLS, K_MAX_WEIGHTS_ROWS};

pub fn normalize_to_symmetric_range(values: &[f64]) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return Err("normalize_to_symmetric_range: empty input".into());
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let span = max_v - min_v;
    if span <= 0.0 {
        return Ok(vec![0.0; values.len()]);
    }
    Ok(values
        .iter()
        .map(|&v| 2.0 * (v - min_v) / span - 1.0)
        .collect())
}

pub fn reshape_row_major(flat: &[f64], rows: usize, cols: usize) -> Result<Vec<Vec<f64>>, String> {
    let need = rows
        .checked_mul(cols)
        .ok_or_else(|| "reshape_row_major: overflow".to_string())?;
    if flat.len() != need {
        return Err(format!(
            "reshape_row_major: expected {} elements, got {}",
            need,
            flat.len()
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        out.push(flat[start..start + cols].to_vec());
    }
    Ok(out)
}

pub fn extract_submatrix(
    matrix: &[Vec<f64>],
    row_off: usize,
    col_off: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if matrix.is_empty() {
        return Err("extract_submatrix: empty matrix".into());
    }
    let src_cols = matrix[0].len();
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != src_cols {
            return Err(format!("extract_submatrix: ragged row {i}"));
        }
    }
    if row_off + n_rows > matrix.len() {
        return Err("extract_submatrix: row slice out of bounds".into());
    }
    if col_off + n_cols > src_cols {
        return Err("extract_submatrix: column slice out of bounds".into());
    }
    if n_rows > K_MAX_WEIGHTS_ROWS as usize {
        return Err("extract_submatrix: row count exceeds k_max_weights_rows".into());
    }
    if n_cols > K_MAX_WEIGHTS_COLS as usize {
        return Err("extract_submatrix: column count exceeds k_max_weights_cols".into());
    }
    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        out.push(matrix[row_off + r][col_off..col_off + n_cols].to_vec());
    }
    Ok(out)
}

pub fn prepare_weight_matrix(
    flat: &[f64],
    rows: usize,
    cols: usize,
    row_off: usize,
    col_off: usize,
    out_rows: usize,
    out_cols: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let full = reshape_row_major(flat, rows, cols)?;
    let sliced = extract_submatrix(&full, row_off, col_off, out_rows, out_cols)?;
    let mut normalized_rows = Vec::with_capacity(out_rows);
    for row in sliced {
        let norm_row = normalize_to_symmetric_range(&row)?;
        normalized_rows.push(norm_row);
    }
    Ok(normalized_rows)
}
