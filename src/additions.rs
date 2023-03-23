use std::error::Error;
use std::ops::Add;
use crate::Matrix;

/// Add two matrices together
///
/// # Examples
///
/// ```
/// use matrix_operations::additions::add;
/// use matrix_operations::Matrix;
///
/// let shape = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape).unwrap();
/// let matrix2 = Matrix::new(data2, shape).unwrap();
///
/// let new_matrix = add(&matrix1, &matrix2).unwrap();
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
/// use matrix_operations::additions::add;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let shape2 = (3, 2);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn add<T: Copy + Add<Output = T> + Default>(terms1: &Matrix<T>, terms2: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
    if terms1.shape != terms2.shape {
        return Err("Matrix shapes are not compatible for addition".into());
    }
    let mut result = Matrix::default(terms1.shape);
    for i in 0..terms1.data.len() {
        result.data[i] = terms1.data[i] + terms2.data[i];
    }
    Ok(result)
}

/// Add a column from a matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::additions::add_column;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (1, 3);
/// let data2 = vec![1, 2, 3];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_column(&matrix1, &matrix2).unwrap();
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
/// If the second matrix is not row with same number of column as the first matrix, an error will be returned
///
/// ```
/// use matrix_operations::additions::add_column;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (1, 2);
/// let data2 = vec![1, 2];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_column(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
///
/// ```
/// use matrix_operations::additions::add_column;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (2, 3);
/// let data2 = vec![1, 2, 3, 4, 5, 6];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_column(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn add_column<T: Copy + Add<Output = T> + Default>(terms1: &Matrix<T>, terms2: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
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

/// Add a row from a matrix
///
/// # Examples
///
/// ```
/// use matrix_operations::additions::add_row;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (2, 1);
/// let data2 = vec![1, 2];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_row(&matrix1, &matrix2).unwrap();
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
/// If the second matrix is not column with same number of row as the first matrix, an error will be returned
///
/// ```
/// use matrix_operations::additions::add_row;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (2, 2);
/// let data2 = vec![1, 2, 3, 4];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_row(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
///
/// ```
/// use matrix_operations::additions::add_row;
/// use matrix_operations::Matrix;
///
/// let shape1 = (2, 3);
/// let data1 = vec![1, 2, 3, 4, 5, 6];
/// let shape2 = (3, 1);
/// let data2 = vec![1, 2, 3];
/// let matrix1 = Matrix::new(data1, shape1).unwrap();
/// let matrix2 = Matrix::new(data2, shape2).unwrap();
///
/// let new_matrix = add_row(&matrix1, &matrix2);
///
/// assert!(new_matrix.is_err());
/// ```
pub fn add_row<T: Copy + Add<Output = T> + Default>(terms1: &Matrix<T>, terms2: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
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
/// use matrix_operations::additions::add_scalar;
/// use matrix_operations::Matrix;
///
/// let shape = (2, 3);
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let matrix = Matrix::new(data, shape).unwrap();
///
/// let new_matrix = add_scalar(&matrix, 2);
///
/// assert_eq!(new_matrix[0][0], 3);
/// assert_eq!(new_matrix[0][1], 4);
/// assert_eq!(new_matrix[0][2], 5);
/// assert_eq!(new_matrix[1][0], 6);
/// assert_eq!(new_matrix[1][1], 7);
/// assert_eq!(new_matrix[1][2], 8);
/// ```
pub fn add_scalar<T: Copy + Add<Output = T> + Default>(terms: &Matrix<T>, scalar: T) -> Matrix<T> {
    let mut result = Matrix::default(terms.shape);
    for i in 0..terms.data.len() {
        result.data[i] = terms.data[i] + scalar;
    }
    result
}
