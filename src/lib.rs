//! # Matrix_Operations
//!
//! Matrix_Operations is a Rust crate for performing matrix operations. It provides a set of functions for performing common matrix operations.
//!
//! ## Installation
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! matrix_operations = "0.1.0"
//! ```
//!
//! ## Usage
//!
//! ## Features
//!
//! - Create a matrix
//! - Transpose a matrix
//! - Multiply / Add / Subtract two matrices
//! - Multiply / Divide / Add / Subtract a matrix by a scalar
//! - Add / Subtract each matrix rows / columns by a distinct value
//! - Apply a function to each element of a matrix (like multiplying by a scalar, or adding a constant)
//! - Apply a function on each element of two matrices (like multiplying two matrices element by element)
//! - Apply a function on each row or column of a matrix
//! - Get a matrix as a slice
//!

pub mod operations;
pub mod csv;
mod macro_matrix;
mod add_impl;
mod sub_impl;
mod mul_impl;

use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Div, Index, IndexMut, Range};
use crate::operations::*;

/// A matrix struct that can be used to perform matrix operations.
pub struct Matrix<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: (usize, usize),
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    /// Allows the matrix to be indexed as a 2-dimensional array
    ///
    /// # Examples
    ///
    /// To get specific elements of the matrix, use the `[row][col]` operator:
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// assert_eq!(matrix[0][0], 1);
    /// assert_eq!(matrix[0][1], 2);
    /// assert_eq!(matrix[0][2], 3);
    /// assert_eq!(matrix[1][0], 4);
    /// assert_eq!(matrix[1][1], 5);
    /// assert_eq!(matrix[1][2], 6);
    /// ```
    ///
    /// To get specific rows of the matrix, use the `[row]` operator:
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// assert_eq!(matrix[0], vec![1, 2, 3]);
    /// assert_eq!(matrix[1], vec![4, 5, 6]);
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.shape.1;
        let end = start + self.shape.1;
        &self.data[start..end]
    }
}

impl<T: Copy + Default> Index<Range<usize>> for Matrix<T> {
    type Output = [T];

    /// Allows the matrix get a range of rows
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2], [3, 4], [5, 6]];
    ///
    /// assert_eq!(matrix[0..1], [1, 2]);
    /// assert_eq!(matrix[1..3], [3, 4, 5, 6]);
    /// ```
    ///
    /// You can convert to a new matrix using the `from_slice` function:
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1 = matrix![[1, 2], [3, 4], [5, 6]];
    ///
    /// let matrix2 = Matrix::from_slice(&matrix1[0..2], (2, matrix1.shape().1)).unwrap();
    ///
    /// assert_eq!(matrix2, matrix![[1, 2], [3, 4]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds
    ///
    /// ```should_panic
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2], [3, 4], [5, 6]];
    ///
    /// // This will panic
    /// let matrix2 = &matrix1[0..4];
    ///
    /// ```
    fn index(&self, index: Range<usize>) -> &Self::Output {
        let start = index.start * self.shape.1;
        let end = index.end * self.shape.1;
        &self.data[start..end]
    }
}

impl<T: Copy> Clone for Matrix<T> where T: Clone {

    /// Allows the matrix to be cloned
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    /// let matrix2 = matrix1.clone();
    ///
    /// assert_eq!(matrix1, matrix2);
    /// ```
    fn clone(&self) -> Self {
        Matrix {
            data: self.data.clone(),
            shape: self.shape,
        }
    }
}

impl<T> PartialEq for Matrix<T> where T: Eq {

    /// Allows the matrix to be compared to another matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// assert_eq!(matrix1, matrix2);
    /// ```
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2, 3],
    ///                       [4, 5, 7]];
    ///
    /// assert_ne!(matrix1, matrix2);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl<T> IndexMut<usize> for Matrix<T> {

    /// Allows the matrix to be modified as a 2-dimensional array
    ///
    /// # Examples
    ///
    /// To modify specific elements of the matrix, use the `[row][col]` operator:
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix[0][0] = 10;
    ///
    /// assert_eq!(matrix[0][0], 10);
    /// ```
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.shape.1;
        let end = start + self.shape.1;
        &mut self.data[start..end]
    }
}

impl<T: Debug> Debug for Matrix<T> {

    /// Allows the matrix to be printed as debug
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// assert_eq!(format!("{:?}", matrix), "(2, 3) : [[1, 2, 3], [4, 5, 6]]");
    /// ```
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from(format!("({}, {}) : ", self.shape.0, self.shape.1));
        s.push_str("[");
        for i in 0..self.shape.0 {
            s.push_str("[");
            for j in 0..self.shape.1 {
                s.push_str(&format!("{:?}, ", self[i][j]));
            }
            s.remove(s.len() - 1);
            s.remove(s.len() - 1);
            s.push_str("], ");
        }
        s.remove(s.len() - 1);
        s.remove(s.len() - 1);
        s.push_str("]");
        write!(f, "{}", s)
    }
}

impl<T> Display for Matrix<T> where T: Display {

    /// Allows the matrix to be printed
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// println!("{}", matrix);
    /// assert_eq!(format!("{}", matrix), "1 2 3 \n4 5 6 \n");
    /// ```
    ///
    /// # Results
    ///
    /// ```text
    /// 1 2 3
    /// 4 5 6
    /// ```
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                s.push_str(&format!("{} ", self[i][j]));
            }
            s.push_str("\n");
        }
        write!(f, "{}", s)
    }
}

impl<T: Copy + Default + Div<Output = T>> Div<T> for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix1 / 2;
    ///
    /// assert_eq!(matrix2, matrix![[0, 1, 1], [2, 2, 3]]);
    /// ```
    fn div(self, scalar: T) -> Self::Output {
        div_matrix_with_scalar(&self, scalar)
    }
}

impl<T: Default + Copy> Matrix<T> {

    /// Creates a new matrix from a 1 dimensional vector and a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// assert_eq!(matrix[0][0], 1);
    /// assert_eq!(matrix[0][1], 2);
    /// assert_eq!(matrix[0][2], 3);
    /// assert_eq!(matrix[1][0], 4);
    /// assert_eq!(matrix[1][1], 5);
    /// assert_eq!(matrix[1][2], 6);
    /// ```
    ///
    /// # Errors
    ///
    /// If the data length does not match the shape, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 2);
    /// let matrix = Matrix::new(data, shape);
    ///
    /// assert!(matrix.is_err());
    /// ```
    pub fn new(data: Vec<T>, shape: (usize, usize)) -> Result<Matrix<T>, Box<dyn Error>> {
        if data.len() != shape.0 * shape.1 {
            return Err("Data length does not match shape".into());
        }
        Ok(Matrix { data, shape })
    }

    /// Creates a new matrix from a shape and initialises all elements with the default value
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    /// let matrix: Matrix<u32> = Matrix::new_default(shape);
    ///
    /// assert_eq!(matrix[0][0], 0);
    /// assert_eq!(matrix[0][1], 0);
    /// assert_eq!(matrix[0][2], 0);
    /// assert_eq!(matrix[1][0], 0);
    /// assert_eq!(matrix[1][1], 0);
    /// assert_eq!(matrix[1][2], 0);
    /// ```
    pub fn new_default(shape: (usize, usize)) -> Matrix<T> {
        Matrix {
            data: vec![T::default(); shape.0 * shape.1],
            shape,
        }
    }

    /// Creates a new matrix from a shape and initialises all elements with a given value
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    /// let matrix = Matrix::new_initialised(shape, 10);
    ///
    /// assert_eq!(matrix[0][0], 10);
    /// assert_eq!(matrix[0][1], 10);
    /// assert_eq!(matrix[0][2], 10);
    /// assert_eq!(matrix[1][0], 10);
    /// assert_eq!(matrix[1][1], 10);
    /// assert_eq!(matrix[1][2], 10);
    /// ```
    pub fn new_initialised(shape: (usize, usize), value: T) -> Matrix<T> {
        Matrix {
            data: vec![value; shape.0 * shape.1],
            shape,
        }
    }

    /// Creates a new matrix from a 1 dimensional slice and a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::from_slice(&data, shape).unwrap();
    ///
    /// assert_eq!(matrix[0][0], 1);
    /// assert_eq!(matrix[0][1], 2);
    /// assert_eq!(matrix[0][2], 3);
    /// assert_eq!(matrix[1][0], 4);
    /// assert_eq!(matrix[1][1], 5);
    /// assert_eq!(matrix[1][2], 6);
    /// ```
    ///
    /// # Errors
    ///
    /// If the data length does not match the shape, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 2);
    /// let matrix = Matrix::from_slice(&data, shape);
    ///
    /// assert!(matrix.is_err());
    /// ```
    pub fn from_slice(data: &[T], shape: (usize, usize)) -> Result<Matrix<T>, Box<dyn Error>> {
        if data.len() != shape.0 * shape.1 {
            return Err("Data length does not match shape".into());
        }
        Ok(Matrix {
            data: data.to_vec(),
            shape,
        })
    }

    /// Creates a new matrix from a 1 dimensional vector and a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// assert_eq!(matrix[0][0], 1);
    /// assert_eq!(matrix[0][1], 2);
    /// assert_eq!(matrix[0][2], 3);
    /// assert_eq!(matrix[1][0], 4);
    /// assert_eq!(matrix[1][1], 5);
    /// assert_eq!(matrix[1][2], 6);
    /// ```
    ///
    /// # Errors
    ///
    /// If the data length does not match the shape, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 2);
    /// let matrix = Matrix::new(data, shape);
    ///
    /// assert!(matrix.is_err());
    /// ```
    pub fn from_vec(data: Vec<T>, shape: (usize, usize)) -> Result<Matrix<T>, Box<dyn Error>> {
        if data.len() != shape.0 * shape.1 {
            return Err("Data length does not match shape".into());
        }
        Ok(Matrix { data, shape })
    }

    /// Creates a new matrix from a 2 dimensional vector
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let matrix = Matrix::from_2d_vec(data).unwrap();
    ///
    /// assert_eq!(matrix[0][0], 1);
    /// assert_eq!(matrix[0][1], 2);
    /// assert_eq!(matrix[0][2], 3);
    /// assert_eq!(matrix[1][0], 4);
    /// assert_eq!(matrix[1][1], 5);
    /// assert_eq!(matrix[1][2], 6);
    /// ```
    ///
    /// # Errors
    ///
    /// If the data vector is not rectangular, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![vec![1, 2, 3], vec![4, 5]];
    /// let matrix = Matrix::from_2d_vec(data);
    ///
    /// assert!(matrix.is_err());
    /// ```
    pub fn from_2d_vec(data: Vec<Vec<T>>) -> Result<Matrix<T>, Box<dyn Error>> {
        let shape = (data.len(), data[0].len());
        for i in 1..shape.0 {
            if data[i].len() != shape.1 {
                return Err("Data length does not match shape".into());
            }
        }
        let mut matrix = Matrix::new_default(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                matrix[i][j] = data[i][j];
            }
        }
        Ok(matrix)
    }

    /// Returns the matrix as a 1 dimensional array
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// assert_eq!(matrix.as_slice(), [1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the matrix as a 1 dimensional vector
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// assert_eq!(matrix.as_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn as_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    /// Returns the matrix as a 2 dimensional vector
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// assert_eq!(matrix.as_2d_vec(), vec![vec![1, 2, 3], vec![4, 5, 6]]);
    /// ```
    pub fn as_2d_vec(&self) -> Vec<Vec<T>> {
        let mut vec = Vec::new();
        for i in 0..self.shape.0 {
            let mut row = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self[i][j]);
            }
            vec.push(row);
        }
        vec
    }

    /// Returns the shape of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// assert_eq!(matrix.shape(), (2, 3));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the column of the matrix at the given index as a Vec
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let column = matrix.get_column(0).unwrap();
    ///
    /// assert_eq!(column, vec![1, 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the column index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let column = matrix.get_column(3);
    ///
    /// assert!(column.is_err());
    /// ```
    pub fn get_column(&self, column: usize) -> Result<Vec<T>, Box<dyn Error>> {
        if column >= self.shape.1 {
            return Err("Column index out of bounds".into());
        }
        let mut column_vec = Vec::new();
        for i in 0..self.shape.0 {
            column_vec.push(self.data[i * self.shape.1 + column]);
        }
        Ok(column_vec)
    }

    /// Returns a matrix with the columns specified by the start and end index
    /// The start index is inclusive and the end index is exclusive
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let columns = matrix.get_columns(0, 2).unwrap();
    ///
    /// assert_eq!(columns, matrix![[1, 2], [4, 5]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the start or end index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let columns = matrix.get_columns(0, 4);
    ///
    /// assert!(columns.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let columns = matrix.get_columns(4, 2);
    ///
    /// assert!(columns.is_err());
    /// ```
    pub fn get_columns(&self, start: usize, end: usize) -> Result<Matrix<T>, Box<dyn Error>> {
        if start >= self.shape.1 || end >= self.shape.1 {
            return Err("Column index out of bounds".into());
        }
        let mut data = Vec::new();
        for i in 0..self.shape.0 {
            for j in start..end {
                data.push(self.data[i * self.shape.1 + j]);
            }
        }
        Ok(Matrix::new(data, (self.shape.0, end - start))?)
    }

    /// Returns the row of the matrix at the given index as a Vec
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let row = matrix.get_row(0).unwrap();
    ///
    /// assert_eq!(row, vec![1, 2, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the row index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let row = matrix.get_row(2);
    ///
    /// assert!(row.is_err());
    /// ```
    pub fn get_row(&self, row: usize) -> Result<Vec<T>, Box<dyn Error>> {
        if row >= self.shape.0 {
            return Err("Row index out of bounds".into());
        }
        Ok(self[row].to_vec())
    }

    /// Returns a matrix with the rows specified by the start and end index
    /// The start index is inclusive and the end index is exclusive
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let rows = matrix.get_rows(0, 1).unwrap();
    ///
    /// assert_eq!(rows, matrix![[1, 2, 3]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the start or end index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let rows = matrix.get_rows(0, 3);
    ///
    /// assert!(rows.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    ///
    /// let rows = matrix.get_rows(3, 4);
    ///
    /// assert!(rows.is_err());
    /// ```
    pub fn get_rows(&self, start: usize, end: usize) -> Result<Matrix<T>, Box<dyn Error>> {
        if start >= self.shape.0 || end > self.shape.0 {
            return Err("Row index out of bounds".into());
        }
        let mut data = Vec::new();
        for i in start..end {
            data.extend_from_slice(&self[i]);
        }
        Ok(Matrix::new(data, (end - start, self.shape.1))?)
    }

    /// Add a column to the matrix
    /// The column will be added to the index specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.add_column(0).unwrap();
    /// assert_eq!(matrix, matrix![[0, 1, 2, 3], [0, 4, 5, 6]]);
    ///
    /// matrix.add_column(matrix.shape().1).unwrap();
    /// assert_eq!(matrix, matrix![[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// matrix.add_column(0).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[0]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the column index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.add_column(4);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_column(&mut self, index: usize) -> Result<(), Box<dyn Error>> {
        if index > self.shape.1 {
            return Err("Column index out of bounds".into());
        }
        if self.shape.0 == 0 {
            self.shape.0 = 1;
            self.shape.1 = 1;
            self.data.push(T::default());
            return Ok(());
        }
        self.shape.1 += 1;
        for i in 0..self.shape.0 {
            self.data.insert(i * self.shape.1 + index, T::default());
        }
        Ok(())
    }

    /// Add a column to the matrix
    /// The column will be added to the index specified
    /// The values of the column will be initialized to the value specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.add_column_initialized(0, 10).unwrap();
    /// assert_eq!(matrix, matrix![[10, 1, 2, 3], [10, 4, 5, 6]]);
    ///
    /// matrix.add_column_initialized(matrix.shape().1, 20).unwrap();
    /// assert_eq!(matrix, matrix![[10, 1, 2, 3, 20], [10, 4, 5, 6, 20]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// matrix.add_column_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[10]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the column index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.add_column_initialized(4, 10);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_column_initialized(&mut self, index: usize, value: T) -> Result<(), Box<dyn Error>> {
        if index > self.shape.1 {
            return Err("Column index out of bounds".into());
        }
        if self.shape.0 == 0 {
            self.shape.0 = 1;
            self.shape.1 = 1;
            self.data.push(value);
            return Ok(());
        }
        self.shape.1 += 1;
        for i in 0..self.shape.0 {
            self.data.insert((i * self.shape.1) + index, value);
        }
        Ok(())
    }

    /// Add a column to the matrix
    /// The column will be added to the index specified
    /// The values of the column will be initialized to the values specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.add_column_from_vec(0, vec![10, 20]).unwrap();
    /// assert_eq!(matrix, matrix![[10, 1, 2, 3], [20, 4, 5, 6]]);
    ///
    /// matrix.add_column_from_vec(matrix.shape().1, vec![30, 40]).unwrap();
    /// assert_eq!(matrix, matrix![[10, 1, 2, 3, 30], [20, 4, 5, 6, 40]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// matrix.add_column_from_vec(0, vec![10, 20]).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[10], [20]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the column index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.add_column_from_vec(4, vec![10, 20]);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_column_from_vec(&mut self, index: usize, values: Vec<T>) -> Result<(), Box<dyn Error>> {
        if index > self.shape.1 {
            return Err("Column index out of bounds".into());
        }
        if self.shape.0 == 0 {
            self.shape.0 = values.len();
            self.shape.1 = 1;
            self.data = values;
            return Ok(());
        }
        if values.len() != self.shape.0 {
            return Err("Values length does not match matrix height".into());
        }
        self.shape.1 += 1;
        for i in 0..self.shape.0 {
            self.data.insert((i * self.shape.1) + index, values[i]);
        }
        Ok(())
    }

    /// Add all matrix column to the first matrix
    /// The column will be added to the end of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let mut matrix2 = matrix![[10, 20],
    ///                          [30, 40]];
    ///
    /// matrix.add_columns_from_matrix(&matrix2).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3, 10, 20], [4, 5, 6, 30, 40]]);
    /// ```
    ///
    /// If the first matrix is empty, a new matrix will be created with the matrix data
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// let matrix2 = matrix![[10], [20]];
    ///
    /// matrix.add_columns_from_matrix(&matrix2).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[10], [20]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the number of rows does not match, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[10, 20, 30]];
    ///
    /// let result = matrix.add_columns_from_matrix(&matrix2);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_columns_from_matrix(&mut self, matrix: &Matrix<T>) -> Result<(), Box<dyn Error>> {
        if self.shape.0 == 0 {
            self.shape.0 = matrix.shape.0;
            self.shape.1 = matrix.shape.1;
            self.data = matrix.data.clone();
            return Ok(());
        }
        if self.shape.0 != matrix.shape().0 {
            return Err("Matrix rows does not match".into());
        }
        self.shape.1 += matrix.shape().1;
        for i in 0..self.shape.0 {
            for j in 0..matrix.shape().1 {
                self.data.insert((i * self.shape.1) + self.shape.1 - matrix.shape().1 + j, matrix.data[i * matrix.shape().1 + j]);
            }
        }
        Ok(())
    }

    /// Remove a column from the matrix
    /// The column will be removed from the index specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.remove_column(1).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[1, 3], [4, 6]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the column index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.remove_column(3);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn remove_column(&mut self, index: usize) -> Result<(), Box<dyn Error>> {
        if index >= self.shape.1 {
            return Err("Column index out of bounds".into());
        }
        self.shape.1 -= 1;
        for i in 0..self.shape.0 {
            self.data.remove((i * self.shape.1) + index);
        }
        Ok(())
    }

    /// Add a row to the matrix
    /// The row will be added to the index specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.add_row(1).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0], [4, 5, 6]]);
    ///
    /// matrix.add_row(matrix.shape().0).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// matrix.add_row(0).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[0]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the row index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.add_row(3);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_row(&mut self, index: usize) -> Result<(), Box<dyn Error>> {
        if index > self.shape.0 {
            return Err("Row index out of bounds".into());
        }
        if self.shape.0 == 0 {
            self.shape.0 = 1;
            self.shape.1 = 1;
            self.data.push(T::default());
            return Ok(());
        }
        self.shape.0 += 1;
        for _ in 0..self.shape.1 {
            self.data.insert(index * self.shape.1, T::default());
        }
        Ok(())
    }

    /// Add a row to the matrix
    /// The row will be added to the index specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.add_row_initialized(1, 10).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [10, 10, 10], [4, 5, 6]]);
    ///
    /// matrix.add_row_initialized(matrix.shape().0, 20).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [10, 10, 10], [4, 5, 6], [20, 20, 20]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// matrix.add_row_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[10]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the row index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.add_row_initialized(3, 10);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_row_initialized(&mut self, index: usize, value: T) -> Result<(), Box<dyn Error>> {
        if index > self.shape.0 {
            return Err("Row index out of bounds".into());
        }
        if self.shape.0 == 0 {
            self.shape.0 = 1;
            self.shape.1 = 1;
            self.data.push(value);
            return Ok(());
        }
        self.shape.0 += 1;
        for _ in 0..self.shape.1 {
            self.data.insert(index * self.shape.1, value);
        }
        Ok(())
    }

    /// Add a row to the matrix
    /// The row will be added to the index specified
    /// The row will be initialized with the values in the vector
    /// The vector must have the same length as the number of columns in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// matrix.add_row_from_vec(1, vec![11, 12, 13]).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [11, 12, 13], [4, 5, 6]]);
    ///
    /// matrix.add_row_from_vec(matrix.shape().0, vec![21, 22, 23]).unwrap();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [11, 12, 13], [4, 5, 6], [21, 22, 23]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// matrix.add_row_from_vec(0, vec![10, 10, 10]).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[10, 10, 10]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the row index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let result = matrix.add_row_from_vec(3, vec![10, 10, 10]);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_row_from_vec(&mut self, index: usize, values: Vec<T>) -> Result<(), Box<dyn Error>> {
        if index > self.shape.0 {
            return Err("Row index out of bounds".into());
        }
        if self.shape.0 == 0 {
            self.shape.0 = 1;
            self.shape.1 = values.len();
            self.data = values;
            return Ok(());
        }
        self.shape.0 += 1;
        for i in (0..self.shape.1).rev() {
            self.data.insert(index * self.shape.1, values[i]);
        }
        Ok(())
    }

    /// Add rows to the matrix
    /// The rows will be added to the index specified
    /// The rows will be initialized with the values in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[10, 10, 10],
    ///                       [20, 20, 20]];
    ///
    /// matrix.add_rows_from_matrix(&matrix2).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6], [10, 10, 10], [20, 20, 20]]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the rows
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![];
    ///
    /// let matrix2 = matrix![[10, 10, 10],
    ///                       [20, 20, 20]];
    ///
    /// matrix.add_rows_from_matrix(&matrix2).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[10, 10, 10], [20, 20, 20]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the matrices do not have the same number of columns, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[10, 10],
    ///                       [20, 20]];
    ///
    /// let result = matrix.add_rows_from_matrix(&matrix2);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn add_rows_from_matrix(&mut self, matrix: &Matrix<T>) -> Result<(), Box<dyn Error>> {
        if self.shape.1 != matrix.shape().1 && self.shape != (0, 0)  {
            return Err("Matrices must have the same number of columns".into());
        }
        for i in 0..matrix.shape().0 {
            self.add_row_from_vec(self.shape.0, matrix.get_row(i).unwrap().to_vec())?;
        }
        Ok(())
    }

    /// Remove a row from the matrix
    /// The row will be removed from the index specified
    /// The shape of the matrix will be updated
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6],
    ///                          [10, 10, 10],
    ///                          [20, 20, 20]];
    ///
    /// matrix.remove_row(2).unwrap();
    ///
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6], [20, 20, 20]]);
    /// ```
    ///
    /// # Errors
    ///
    /// If the row index is out of bounds, an error will be returned
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6],
    ///                          [10, 10, 10],
    ///                          [20, 20, 20]];
    ///
    /// let result = matrix.remove_row(4);
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn remove_row(&mut self, index: usize) -> Result<(), Box<dyn Error>> {
        if index >= self.shape.0 {
            return Err("Row index out of bounds".into());
        }
        self.shape.0 -= 1;
        for _ in 0..self.shape.1 {
            self.data.remove(index * self.shape.1);
        }
        Ok(())
    }

    /// Change the type of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{Matrix, matrix};
    ///
    /// let mut matrix: Matrix<u16> = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let new_matrix: Matrix<f32> = matrix.as_type::<f32>();
    ///
    /// assert_eq!(new_matrix[0], [1.0, 2.0, 3.0]);
    /// assert_eq!(new_matrix[1], [4.0, 5.0, 6.0]);
    /// ```
    pub fn as_type<U: From<T> + Copy + Default>(&mut self) -> Matrix<U> {
        let mut data: Vec<U> = Vec::new();
        for i in 0..self.data.len() {
            data.push(U::from(self.data[i]));
        }
        Matrix::new(data, self.shape).unwrap()
    }
}
