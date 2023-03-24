//! This module will perform the CSV parsing and writing.

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use crate::Matrix;

/// Load a CSV file into a Matrix.
///
/// # Examples
///
/// ```
/// use matrix_operations::csv::load_matrix_from_csv;
///
/// let path = "resources/test.csv";
/// let separator = ",";
///
/// let matrix = load_matrix_from_csv::<f32>(path, separator).unwrap();
///
/// assert_eq!(matrix.shape(), (3, 2));
/// assert_eq!(matrix[0], vec![1.0, 2.0]);
/// assert_eq!(matrix[1], vec![3.0, 4.0]);
/// assert_eq!(matrix[2], vec![5.0, 6.0]);
/// ```
///
/// # Panics
///
/// This function will panic if the file cannot be opened or if the separator is not found.
///
/// ```should_panic
/// use matrix_operations::csv::load_matrix_from_csv;
///
/// let path = "../resources/test_does_not_exist.csv";
/// let separator = ",";
///
/// let matrix = load_matrix_from_csv::<f32>(path, separator).unwrap();
/// ```
pub fn load_matrix_from_csv<T: Default + Copy + FromStr>(path: &str, separator: &str) -> Result<Matrix<T>, Box<dyn Error>> where T: Default + Copy + FromStr, <T as FromStr>::Err: 'static, <T as FromStr>::Err: Error {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let data: Vec<T> = Vec::new();
    let mut matrix = Matrix::new(data, (0, 0))?;

    let mut header = true;
    for line in reader.lines() {
        if header {
            header = false;
            continue;
        }
        let line_tmp = line?;
        let split: Vec<&str> = line_tmp.split(separator).collect();
        let mut row: Vec<T> = Vec::new();
        for i in 0..split.len() {
            let value = split[i].parse::<T>()?;
            row.push(value);
        }
        matrix.add_row_from_vec(matrix.shape.0, row)?;
    }

    Ok(matrix)
}