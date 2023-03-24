//! This module contains functions for matrix operations

use std::error::Error;
use std::iter::Map;
use std::ops::{Add, Div};
use crate::Matrix;

/// Transpose the matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::transpose_matrix;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let shape = (2, 3);
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// let new_matrix = transpose_matrix(&matrix);
///
/// assert_eq!(new_matrix[0][0], 1);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[1][0], 2);
/// assert_eq!(new_matrix[1][1], 5);
/// assert_eq!(new_matrix[2][0], 3);
/// assert_eq!(new_matrix[2][1], 6);
/// assert_eq!(new_matrix.shape(), (3, 2));
/// ```
pub fn transpose_matrix<T: Copy + Default>(matrix: &Matrix<T>) -> Matrix<T> {
    let mut result = Matrix::default((matrix.shape.1, matrix.shape.0));
    for i in 0..matrix.shape.0 {
        for j in 0..matrix.shape.1 {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

/// Add two matrices together
///
/// # Examples
///
/// ```
/// use matrix_operations::operations::add_matrices;
/// use matrix_operations::Matrix;
///
/// let shape = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape).unwrap();
/// let matrix2 = Matrix::new(data2, shape).unwrap();
///
/// let new_matrix = add_matrices(&matrix1, &matrix2).unwrap();
///
/// assert_eq!(new_matrix[0][0], 2);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[0][2], 6);
/// assert_eq!(new_matrix[1][0], 8);
/// assert_eq!(new_matrix[1][1], 10);
/// assert_eq!(new_matrix[1][2], 12);
/// ```
///
/// # Errors
///
/// If the matrices are not compatible for addition, an error will be returned
///
/// ```
/// use matrix_operations::operations::add_matrices;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let shape2 = (3, 2);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrices(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn add_matrices<T: Copy + Add<Output = T> + Default>(terms1: &Matrix<T>, terms2: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
    if terms1.shape != terms2.shape {
        return Err("Matrix shapes are not compatible for addition".into());
    }
    let mut result = Matrix::default(terms1.shape);
    for i in 0..terms1.data.len() {
        result.data[i] = terms1.data[i] + terms2.data[i];
    }
    Ok(result)
}

/// Perform element-wise addition of a matrix and a 1 row matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::operations::add_matrix_with_1row_matrix;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (1, 3);
/// let data2 = vec![1, 2, 3];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrix_with_1row_matrix(&matrix1, &matrix2).unwrap();
///
/// assert_eq!(new_matrix[0][0], 2);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[0][2], 6);
/// assert_eq!(new_matrix[1][0], 5);
/// assert_eq!(new_matrix[1][1], 7);
/// assert_eq!(new_matrix[1][2], 9);
/// ```
///
/// # Errors
///
/// If the second matrix is not a row with same number of column as the first matrix, an error will be returned
///
/// ```
/// use matrix_operations::operations::add_matrix_with_1row_matrix;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (1, 2);
/// let data2 = vec![1, 2];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrix_with_1row_matrix(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
///
/// ```
/// use matrix_operations::operations::add_matrix_with_1row_matrix;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (2, 3);
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrix_with_1row_matrix(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn add_matrix_with_1row_matrix<T: Copy + Add<Output = T> + Default>(terms1: &Matrix<T>, terms2: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
    if terms2.shape.0 != 1 {
        return Err("Second matrix need to have 1 row".into());
    }
    if terms2.shape.1 != terms1.shape.1 {
        return Err("Second matrix need to have same number of columns".into());
    }
    let mut result = Matrix::default(terms1.shape);
    for i in 0..terms1.data.len() {
        result.data[i] = terms1.data[i] + terms2.data[i % terms2.shape.1];
    }
    Ok(result)
}

/// Perform element-wise addition of a matrix and a 1 column matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::operations::add_matrix_with_1col_matrix;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (2, 1);
/// let data2 = vec![1, 2];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrix_with_1col_matrix(&matrix1, &matrix2).unwrap();
///
/// assert_eq!(new_matrix[0][0], 2);
/// assert_eq!(new_matrix[0][1], 3);
/// assert_eq!(new_matrix[0][2], 4);
/// assert_eq!(new_matrix[1][0], 6);
/// assert_eq!(new_matrix[1][1], 7);
/// assert_eq!(new_matrix[1][2], 8);
/// ```
///
/// # Errors
///
/// If the second matrix is not a column with same number of row as the first matrix, an error will be returned
///
/// ```
/// use matrix_operations::operations::add_matrix_with_1col_matrix;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (2, 2);
/// let data2 = vec![1, 2, 3, 4];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrix_with_1col_matrix(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
///
/// ```
/// use matrix_operations::operations::add_matrix_with_1col_matrix;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (3, 1);
/// let data2 = vec![1, 2, 3];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_matrix_with_1col_matrix(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn add_matrix_with_1col_matrix<T: Copy + Add<Output = T> + Default>(terms1: &Matrix<T>, terms2: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
    if terms2.shape.1 != 1 {
        return Err("Second matrix need to have 1 column".into());
    }
    if terms2.shape.0 != terms1.shape.0 {
        return Err("Second matrix need to have same number of rows".into());
    }
    let mut result = Matrix::default(terms1.shape);
    for i in 0..terms1.data.len() {
        result.data[i] = terms1.data[i] + terms2.data[i / terms1.shape.1];
    }
    Ok(result)
}

/// Add a scalar from a matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::operations::add_matrix_with_scalar;
/// use matrix_operations::Matrix;
///
/// let shape = (2, 3);
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// let new_matrix = add_matrix_with_scalar(&matrix, 2);
///
/// assert_eq!(new_matrix[0][0], 3);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[0][2], 5);
/// assert_eq!(new_matrix[1][0], 6);
/// assert_eq!(new_matrix[1][1], 7);
/// assert_eq!(new_matrix[1][2], 8);
/// ```
pub fn add_matrix_with_scalar<T: Copy + Add<Output = T> + Default>(terms: &Matrix<T>, scalar: T) -> Matrix<T> {
    let mut result = Matrix::default(terms.shape);
    for i in 0..terms.data.len() {
        result.data[i] = terms.data[i] + scalar;
    }
    result
}

/// Apply a function to each element of the matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_to_matrix;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let shape = (2, 3);
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// let new_matrix = apply_to_matrix(&matrix, |x| x * 2);
///
/// assert_eq!(new_matrix[0][0], 2);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[0][2], 6);
/// assert_eq!(new_matrix[1][0], 8);
/// assert_eq!(new_matrix[1][1], 10);
/// assert_eq!(new_matrix[1][2], 12);
/// ```
pub fn apply_to_matrix<T: Copy + Default>(matrix: &Matrix<T>, f: fn(T) -> T) -> Matrix<T> {
    let mut result = Matrix::default(matrix.shape);
    for i in 0..matrix.data.len() {
        result.data[i] = f(matrix.data[i]);
    }
    result
}

/// Apply a function to each element of a column of the matrix and return a new matrix
/// The function takes a Vec of the column elements and returns a Vec of the new column elements
/// The function must return a Vec of the same length as the column
///
/// # Examples
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_to_matrix_columns;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let shape = (2, 3);
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// fn sum_column(column: Vec<i32>) -> Vec<i32> {
///    let sum = column.iter().sum();
///    vec![sum; column.len()]
/// }
///
/// let new_matrix = apply_to_matrix_columns(&matrix, sum_column).unwrap();
///
/// assert_eq!(new_matrix[0][0], 5);
/// assert_eq!(new_matrix[0][1], 7);
/// assert_eq!(new_matrix[0][2], 9);
/// assert_eq!(new_matrix[1][0], 5);
/// assert_eq!(new_matrix[1][1], 7);
/// assert_eq!(new_matrix[1][2], 9);
/// ```
///
/// # Errors
///
/// If the function does not return a Vec of the same length as the column, an error will be returned
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_to_matrix_columns;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let shape = (2, 3);
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// fn sum_column(column: Vec<i32>) -> Vec<i32> {
///    let sum = column.iter().sum();
///    vec![sum]
/// }
///
/// let new_matrix = apply_to_matrix_columns(&matrix, sum_column);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn apply_to_matrix_columns<T: Copy + Default>(matrix: &Matrix<T>, f: fn(Vec<T>) -> Vec<T>) -> Result<Matrix<T>, Box<dyn Error>> {
    let mut result = Matrix::default(matrix.shape);
    for i in 0..matrix.shape.1 {
        let column = matrix.get_column(i).unwrap();
        let new_column = f(column);
        if new_column.len() != matrix.shape.0 {
            return Err("Function did not return a Vec of the same length as the column".into());
        }
        for j in 0..matrix.shape.0 {
            result.data[j * matrix.shape.1 + i] = new_column[j];
        }
    }
    Ok(result)
}

/// Apply a function to each element of a row of the matrix and return a new matrix
/// The function takes a Vec of the row elements and returns a Vec of the new row elements
/// The function must return a Vec of the same length as the row
///
/// # Examples
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_to_matrix_rows;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let shape = (2, 3);
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// fn sum_row(row: Vec<i32>) -> Vec<i32> {
///    let sum = row.iter().sum();
///    vec![sum; row.len()]
/// }
///
/// let new_matrix = apply_to_matrix_rows(&matrix, sum_row).unwrap();
///
/// assert_eq!(new_matrix[0][0], 6);
/// assert_eq!(new_matrix[0][1], 6);
/// assert_eq!(new_matrix[0][2], 6);
/// assert_eq!(new_matrix[1][0], 15);
/// assert_eq!(new_matrix[1][1], 15);
/// assert_eq!(new_matrix[1][2], 15);
/// ```
///
/// # Errors
///
/// If the function does not return a Vec of the same length as the row, an error will be returned
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_to_matrix_rows;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let shape = (2, 3);
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// fn sum_row(row: Vec<i32>) -> Vec<i32> {
///    let sum = row.iter().sum();
///    vec![sum]
/// }
///
/// let new_matrix = apply_to_matrix_rows(&matrix, sum_row);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn apply_to_matrix_rows<T: Copy + Default>(matrix: &Matrix<T>, f: fn(Vec<T>) -> Vec<T>) -> Result<Matrix<T>, Box<dyn Error>> {
    let mut result = Matrix::default(matrix.shape);
    for i in 0..matrix.shape.0 {
        let new_row = f(matrix[i].to_vec());
        if new_row.len() != matrix.shape.1 {
            return Err("Function did not return a Vec of the same length as the row".into());
        }
        for j in 0..matrix.shape.1 {
            result.data[i * matrix.shape.1 + j] = new_row[j];
        }
    }
    Ok(result)
}

/// Apply a function to each element of two matrices and return a new matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_between_matrices;
///
/// let shape = (2, 3);
///
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape).unwrap();
///
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix2 = Matrix::new(data2, shape).unwrap();
///
/// let new_matrix = apply_between_matrices(&matrix1, &matrix2, |x, y| x + y).unwrap();
///
/// assert_eq!(new_matrix[0][0], 2);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[0][2], 6);
/// assert_eq!(new_matrix[1][0], 8);
/// assert_eq!(new_matrix[1][1], 10);
/// assert_eq!(new_matrix[1][2], 12);
/// ```
///
/// # Errors
///
/// If the matrices are not the same shape, an error will be returned
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::apply_between_matrices;
///
/// let shape1 = (2, 3);
/// let shape2 = (3, 2);
///
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
///
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = apply_between_matrices(&matrix1, &matrix2, |x, y| x + y);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn apply_between_matrices<T: Copy + Default>(matrix1: &Matrix<T>, matrix2: &Matrix<T>, f: fn(T, T) -> T) -> Result<Matrix<T>, Box<dyn Error>> {
    if matrix1.shape != matrix2.shape {
        return Err("Matrix shapes are not compatible to apply function".into());
    }
    let mut result = Matrix::default(matrix1.shape);
    for i in 0..matrix1.data.len() {
        result.data[i] = f(matrix1.data[i], matrix2.data[i]);
    }
    Ok(result)
}

/// Divide a scalar from a matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::Matrix;
/// use matrix_operations::operations::div_matrix_with_scalar;
///
/// let shape = (2, 3);
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// let new_matrix = div_matrix_with_scalar(&matrix, 2);
///
/// assert_eq!(new_matrix[0][0], 0);
/// assert_eq!(new_matrix[0][1], 1);
/// assert_eq!(new_matrix[0][2], 1);
/// assert_eq!(new_matrix[1][0], 2);
/// assert_eq!(new_matrix[1][1], 2);
/// assert_eq!(new_matrix[1][2], 3);
/// ```
pub fn div_matrix_with_scalar<T: Default + Copy + Div<Output = T>>(matrix: &Matrix<T>, scalar: T) -> Matrix<T> {
    let mut result = Matrix::default(matrix.shape);
    for i in 0..matrix.data.len() {
        result.data[i] = matrix.data[i] / scalar;
    }
    result
}