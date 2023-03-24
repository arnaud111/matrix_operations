//! This module will perform the CSV parsing and writing.

use std::error::Error;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;
use crate::Matrix;

/// Load a CSV file into a Matrix.
///
/// # Examples
///
/// ```
/// use matrix_operations::csv::load_matrix_from_csv;
///
/// let path = "resources/test_read.csv";
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

/// Write a Matrix to a CSV file.
///
/// # Examples
///
/// ```
/// use matrix_operations::csv::write_matrix_to_csv;
///
/// let matrix = matrix_operations::Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)).unwrap();
/// let path = "resources/test_write.csv";
///
/// write_matrix_to_csv(&matrix, path, ",").unwrap();
/// ```
pub fn write_matrix_to_csv<T: Display>(matrix: &Matrix<T>, path: &str, separator: &str) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    let mut header = String::new();
    for i in 0..matrix.shape.1 {
        header.push_str(&format!("col_{}", i));
        if i != matrix.shape.1 - 1 {
            header.push_str(separator);
        }
    }
    header.push_str("\n");
    file.write_all(header.as_bytes())?;
    for i in 0..matrix.shape.0 {
        let mut row = String::new();
        for j in 0..matrix.shape.1 {
            row.push_str(&format!("{}", matrix[i][j]));
            if j != matrix.shape.1 - 1 {
                row.push_str(separator);
            }
        }
        row.push_str("\n");
        file.write_all(row.as_bytes())?;
    }
    Ok(())
}

/// Write a Matrix to a CSV file with headers.
///
/// # Examples
///
/// ```
/// use matrix_operations::csv::write_matrix_to_csv_with_headers;
///
/// let matrix = matrix_operations::Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)).unwrap();
/// let path = "resources/test_write_headers.csv";
/// let headers = vec!["col_1_header".to_string(), "col_2_header".to_string()];
///
/// write_matrix_to_csv_with_headers(&matrix, path, ",", headers).unwrap();
/// ```
///
/// # Errors
///
/// This function will return an error if the number of headers does not match the number of columns in the matrix.
///
/// ```
/// use matrix_operations::csv::write_matrix_to_csv_with_headers;
///
/// let matrix = matrix_operations::Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)).unwrap();
/// let path = "resources/test_write_headers.csv";
/// let headers = vec!["col_1_header".to_string()];
///
/// let result = write_matrix_to_csv_with_headers(&matrix, path, ",", headers);
///
/// assert!(result.is_err());
/// ```
pub fn write_matrix_to_csv_with_headers<T: Display>(matrix: &Matrix<T>, path: &str, separator: &str, headers: Vec<String>) -> Result<(), Box<dyn Error>> {
    if headers.len() != matrix.shape.1 {
        return Err("The number of headers does not match the number of columns in the matrix.".into());
    }
    let mut file = File::create(path)?;
    let mut header = String::new();
    for i in 0..matrix.shape.1 {
        header.push_str(&format!("{}", headers[i]));
        if i != matrix.shape.1 - 1 {
            header.push_str(separator);
        }
    }
    header.push_str("\n");
    file.write_all(header.as_bytes())?;
    for i in 0..matrix.shape.0 {
        let mut row = String::new();
        for j in 0..matrix.shape.1 {
            row.push_str(&format!("{}", matrix[i][j]));
            if j != matrix.shape.1 - 1 {
                row.push_str(separator);
            }
        }
        row.push_str("\n");
        file.write_all(row.as_bytes())?;
    }
    Ok(())
}