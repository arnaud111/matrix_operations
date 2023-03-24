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
//! - Multiply / Divide / Add / Subtract a matrix each rows / columns by another row / column
//! - Apply a function to each element of a matrix (like multiplying by a scalar, or adding a constant)
//! - Apply a function on each element of two matrices (like multiplying two matrices element by element)
//! - Apply a function on each row or column of a matrix
//! - Get a matrix as a slice
//!

pub mod operations;

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, Div, Index, IndexMut, Mul, Sub};

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
    /// To get specific rows of the matrix, use the `[row]` operator:
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
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

impl<T> IndexMut<usize> for Matrix<T> {

    /// Allows the matrix to be modified as a 2-dimensional array
    ///
    /// # Examples
    ///
    /// To modify specific elements of the matrix, use the `[row][col]` operator:
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let mut data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
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

impl<T> Display for Matrix<T> where T: Display {

    /// Allows the matrix to be printed
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
    /// let matrix: Matrix<u32> = Matrix::default(shape);
    ///
    /// assert_eq!(matrix[0][0], 0);
    /// assert_eq!(matrix[0][1], 0);
    /// assert_eq!(matrix[0][2], 0);
    /// assert_eq!(matrix[1][0], 0);
    /// assert_eq!(matrix[1][1], 0);
    /// assert_eq!(matrix[1][2], 0);
    /// ```
    pub fn default(shape: (usize, usize)) -> Matrix<T> {
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
    /// let matrix = Matrix::initialised(shape, 10);
    ///
    /// assert_eq!(matrix[0][0], 10);
    /// assert_eq!(matrix[0][1], 10);
    /// assert_eq!(matrix[0][2], 10);
    /// assert_eq!(matrix[1][0], 10);
    /// assert_eq!(matrix[1][1], 10);
    /// assert_eq!(matrix[1][2], 10);
    /// ```
    pub fn initialised(shape: (usize, usize), value: T) -> Matrix<T> {
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
        let mut matrix = Matrix::default(shape);
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// assert_eq!(matrix.as_slice(), [1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the shape of the matrix
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
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

    /// Returns the row of the matrix at the given index as a Vec
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
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

    /// Add a column to the matrix
    /// The column will be added to the index specified
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
    /// let new_matrix = matrix.append_column(1).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (2, 4));
    /// assert_eq!(new_matrix.as_slice(), [1, 0, 2, 3, 4, 0, 5, 6]);
    /// ```
    ///
    /// If the index is greater than the number of columns, the column will be added to the end
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_column(3).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (2, 4));
    /// assert_eq!(new_matrix.as_slice(), [1, 2, 3, 0, 4, 5, 6, 0]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_column(0).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (1, 1));
    /// assert_eq!(new_matrix.as_slice(), [0]);
    /// ```
    pub fn append_column(&self, index: usize) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.data.len() == 0 {
            return Matrix::new(vec![T::default()], (1, 1));
        }
        let mut new_data = Vec::new();
        for i in 0..self.data.len() {
            if i % self.shape.1 == index {
                new_data.push(T::default());
            }
            new_data.push(self.data[i]);
            if index > self.shape.1 - 1 && i % self.shape.1 == self.shape.1 - 1 {
                new_data.push(T::default());
            }
        }
        Matrix::new(new_data, (self.shape.0, self.shape.1 + 1))
    }

    /// Add a column to the matrix
    /// The column will be added to the index specified
    /// The values of the column will be initialized to the value specified
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
    /// let new_matrix = matrix.append_column_initialized(1, 10).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (2, 4));
    /// assert_eq!(new_matrix.as_slice(), [1, 10, 2, 3, 4, 10, 5, 6]);
    /// ```
    ///
    /// If the index is greater than the number of columns, the column will be added to the end
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_column_initialized(3, 10).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (2, 4));
    /// assert_eq!(new_matrix.as_slice(), [1, 2, 3, 10, 4, 5, 6, 10]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_column_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (1, 1));
    /// assert_eq!(new_matrix.as_slice(), [10]);
    /// ```
    pub fn append_column_initialized(&self, index: usize, value: T) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.data.len() == 0 {
            return Matrix::new(vec![value], (1, 1));
        }
        let mut new_data = Vec::new();
        for i in 0..self.data.len() {
            if i % self.shape.1 == index {
                new_data.push(value);
            }
            new_data.push(self.data[i]);
            if index > self.shape.1 - 1 && i % self.shape.1 == self.shape.1 - 1 {
                new_data.push(value);
            }
        }
        Matrix::new(new_data, (self.shape.0, self.shape.1 + 1))
    }

    /// Add a row to the matrix
    /// The row will be added to the index specified
    /// If the index is greater than the number of rows, the row will be added to the end
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
    /// let new_matrix = matrix.append_row(1).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (3, 3));
    /// assert_eq!(new_matrix.as_slice(), [1, 2, 3, 0, 0, 0, 4, 5, 6]);
    /// ```
    ///
    /// If the index is greater than the number of rows, the row will be added to the end
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_row(2).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (3, 3));
    /// assert_eq!(new_matrix.as_slice(), [1, 2, 3, 4, 5, 6, 0, 0, 0]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_row(0).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (1, 1));
    /// assert_eq!(new_matrix.as_slice(), [0]);
    /// ```
    pub fn append_row(&self, index: usize) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.data.len() == 0 {
            return Matrix::new(vec![T::default()], (1, 1));
        }
        let mut new_data = Vec::new();
        for i in 0..index * self.shape.1 {
            if i < self.data.len() {
                new_data.push(self.data[i]);
            }
        }
        for _ in 0..self.shape.1 {
            new_data.push(T::default());
        }
        for i in index * self.shape.1..self.data.len() {
            new_data.push(self.data[i]);
        }
        Matrix::new(new_data, (self.shape.0 + 1, self.shape.1))
    }

    /// Add a row to the matrix
    /// The row will be added to the index specified
    /// If the index is greater than the number of rows, the row will be added to the end
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
    /// let new_matrix = matrix.append_row_initialized(1, 10).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (3, 3));
    /// assert_eq!(new_matrix.as_slice(), [1, 2, 3, 10, 10, 10, 4, 5, 6]);
    /// ```
    ///
    /// If the index is greater than the number of rows, the row will be added to the end
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_row_initialized(2, 10).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (3, 3));
    /// assert_eq!(new_matrix.as_slice(), [1, 2, 3, 4, 5, 6, 10, 10, 10]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.append_row_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(new_matrix.shape(), (1, 1));
    /// assert_eq!(new_matrix.as_slice(), [10]);
    /// ```
    pub fn append_row_initialized(&self, index: usize, value: T) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.data.len() == 0 {
            return Matrix::new(vec![value], (1, 1));
        }
        let mut new_data = Vec::new();
        for i in 0..index * self.shape.1 {
            if i < self.data.len() {
                new_data.push(self.data[i]);
            }
        }
        for _ in 0..self.shape.1 {
            new_data.push(value);
        }
        for i in index * self.shape.1..self.data.len() {
            new_data.push(self.data[i]);
        }
        Matrix::new(new_data, (self.shape.0 + 1, self.shape.1))
    }
}

impl<T: Default + Copy + Mul<Output = T> + AddAssign<<T as Mul>::Output>> Matrix<T> {

    /// Multiply two matrices together
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    ///
    /// let shape2 = (3, 2);
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.dot(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 22);
    /// assert_eq!(new_matrix[0][1], 28);
    /// assert_eq!(new_matrix[1][0], 49);
    /// assert_eq!(new_matrix[1][1], 64);
    /// ```
    ///
    /// # Errors
    ///
    /// If the matrices are not compatible for multiplication, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    ///
    /// let shape2 = (2, 3);
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.dot(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn dot(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape.1 != other.shape.0 {
            return Err("Matrix shapes are not compatible for dot product".into());
        }
        let mut matrix = Matrix::default((self.shape.0, other.shape.1));
        let mut row_self;
        let mut col_other;
        for i in 0..matrix.data.len() {
            row_self = i / matrix.shape.1 * self.shape.1;
            col_other = i % matrix.shape.1;
            for j in 0..self.shape.1 {
                matrix.data[i] += self.data[row_self + j] * other.data[j * other.shape.1 + col_other];
            }
        }
        Ok(matrix)
    }
}

impl<T: Default + Copy + Mul<Output = T>> Matrix<T> {

    /// Multiply a column from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (1, 3);
    /// let data2 = vec![1, 2, 3];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.mul_column(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 1);
    /// assert_eq!(new_matrix[0][1], 4);
    /// assert_eq!(new_matrix[0][2], 9);
    /// assert_eq!(new_matrix[1][0], 4);
    /// assert_eq!(new_matrix[1][1], 10);
    /// assert_eq!(new_matrix[1][2], 18);
    /// ```
    ///
    /// # Errors
    ///
    /// If the second matrix is not row with same number of column as the first matrix, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (1, 2);
    /// let data2 = vec![1, 2];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.mul_column(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 3);
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.mul_column(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn mul_column(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if other.shape.0 != 1 {
            return Err("Second matrix need to have 1 row".into());
        }
        if other.shape.1 != self.shape.1 {
            return Err("Second matrix need to have same number of columns".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] * other.data[i % other.shape.1];
        }
        Ok(matrix)
    }

    /// Multiply a row from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 1);
    /// let data2 = vec![1, 2];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.mul_row(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 1);
    /// assert_eq!(new_matrix[0][1], 2);
    /// assert_eq!(new_matrix[0][2], 3);
    /// assert_eq!(new_matrix[1][0], 8);
    /// assert_eq!(new_matrix[1][1], 10);
    /// assert_eq!(new_matrix[1][2], 12);
    /// ```
    ///
    /// # Errors
    ///
    /// If the second matrix is not column with same number of row as the first matrix, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 2);
    /// let data2 = vec![1, 2, 3, 4];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.mul_row(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (3, 1);
    /// let data2 = vec![1, 2, 3];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.mul_row(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn mul_row(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if other.shape.1 != 1 {
            return Err("Second matrix need to have 1 column".into());
        }
        if other.shape.0 != self.shape.0 {
            return Err("Second matrix need to have same number of rows".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] * other.data[i / self.shape.1];
        }
        Ok(matrix)
    }

    /// Multiply a scalar from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.mul_scalar(2);
    ///
    /// assert_eq!(new_matrix[0][0], 2);
    /// assert_eq!(new_matrix[0][1], 4);
    /// assert_eq!(new_matrix[0][2], 6);
    /// assert_eq!(new_matrix[1][0], 8);
    /// assert_eq!(new_matrix[1][1], 10);
    /// assert_eq!(new_matrix[1][2], 12);
    /// ```
    pub fn mul_scalar(&self, scalar: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] * scalar;
        }
        matrix
    }
}

impl<T: Default + Copy + Sub<Output = T>> Matrix<T> {

    /// Subtract two matrices together
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape).unwrap();
    /// let matrix2 = Matrix::new(data2, shape).unwrap();
    ///
    /// let new_matrix = matrix1.sub(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 0);
    /// assert_eq!(new_matrix[0][1], 0);
    /// assert_eq!(new_matrix[0][2], 0);
    /// assert_eq!(new_matrix[1][0], 0);
    /// assert_eq!(new_matrix[1][1], 0);
    /// assert_eq!(new_matrix[1][2], 0);
    /// ```
    ///
    /// # Errors
    ///
    /// If the matrices are not compatible for subtraction, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let shape2 = (3, 2);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn sub(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape != other.shape {
            return Err("Matrix shapes are not compatible for subtraction".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] - other.data[i];
        }
        Ok(matrix)
    }

    /// Subtract a column from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (1, 3);
    /// let data2 = vec![1, 2, 3];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub_column(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 0);
    /// assert_eq!(new_matrix[0][1], 0);
    /// assert_eq!(new_matrix[0][2], 0);
    /// assert_eq!(new_matrix[1][0], 3);
    /// assert_eq!(new_matrix[1][1], 3);
    /// assert_eq!(new_matrix[1][2], 3);
    /// ```
    ///
    /// # Errors
    ///
    /// If the second matrix is not row with same number of column as the first matrix, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (1, 2);
    /// let data2 = vec![1, 2];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub_column(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 3);
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub_column(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn sub_column(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if other.shape.0 != 1 {
            return Err("Second matrix need to have 1 row".into());
        }
        if other.shape.1 != self.shape.1 {
            return Err("Second matrix need to have same number of columns".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] - other.data[i % other.shape.1];
        }
        Ok(matrix)
    }

    /// Subtract a row from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 1);
    /// let data2 = vec![1, 2];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub_row(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 0);
    /// assert_eq!(new_matrix[0][1], 1);
    /// assert_eq!(new_matrix[0][2], 2);
    /// assert_eq!(new_matrix[1][0], 2);
    /// assert_eq!(new_matrix[1][1], 3);
    /// assert_eq!(new_matrix[1][2], 4);
    /// ```
    ///
    /// # Errors
    ///
    /// If the second matrix is not column with same number of row as the first matrix, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 2);
    /// let data2 = vec![1, 2, 3, 4];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub_row(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (3, 1);
    /// let data2 = vec![1, 2, 3];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.sub_row(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn sub_row(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if other.shape.1 != 1 {
            return Err("Second matrix need to have 1 column".into());
        }
        if other.shape.0 != self.shape.0 {
            return Err("Second matrix need to have same number of rows".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] - other.data[i / self.shape.1];
        }
        Ok(matrix)
    }

    /// Subtract a scalar from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.sub_scalar(2);
    ///
    /// assert_eq!(new_matrix[0][0], -1);
    /// assert_eq!(new_matrix[0][1], 0);
    /// assert_eq!(new_matrix[0][2], 1);
    /// assert_eq!(new_matrix[1][0], 2);
    /// assert_eq!(new_matrix[1][1], 3);
    /// assert_eq!(new_matrix[1][2], 4);
    /// ```
    pub fn sub_scalar(&self, scalar: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] - scalar;
        }
        matrix
    }
}

impl<T: Default + Copy + Div<Output = T>> Matrix<T> {

    /// Divide a column from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (1, 3);
    /// let data2 = vec![1, 2, 3];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.div_column(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 1);
    /// assert_eq!(new_matrix[0][1], 1);
    /// assert_eq!(new_matrix[0][2], 1);
    /// assert_eq!(new_matrix[1][0], 4);
    /// assert_eq!(new_matrix[1][1], 2);
    /// assert_eq!(new_matrix[1][2], 2);
    /// ```
    ///
    /// # Errors
    ///
    /// If the second matrix is not row with same number of column as the first matrix, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (1, 2);
    /// let data2 = vec![1, 2];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.div_column(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 3);
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.div_column(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn div_column(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if other.shape.0 != 1 {
            return Err("Second matrix need to have 1 row".into());
        }
        if other.shape.1 != self.shape.1 {
            return Err("Second matrix need to have same number of columns".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] / other.data[i % other.shape.1];
        }
        Ok(matrix)
    }

    /// Divide a row from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 1);
    /// let data2 = vec![1, 2];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.div_row(&matrix2).unwrap();
    ///
    /// assert_eq!(new_matrix[0][0], 1);
    /// assert_eq!(new_matrix[0][1], 2);
    /// assert_eq!(new_matrix[0][2], 3);
    /// assert_eq!(new_matrix[1][0], 2);
    /// assert_eq!(new_matrix[1][1], 2);
    /// assert_eq!(new_matrix[1][2], 3);
    /// ```
    ///
    /// # Errors
    ///
    /// If the second matrix is not column with same number of row as the first matrix, an error will be returned
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (2, 2);
    /// let data2 = vec![1, 2, 3, 4];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.div_row(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let shape2 = (3, 1);
    /// let data2 = vec![1, 2, 3];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.div_row(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn div_row(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if other.shape.1 != 1 {
            return Err("Second matrix need to have 1 column".into());
        }
        if other.shape.0 != self.shape.0 {
            return Err("Second matrix need to have same number of rows".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] / other.data[i / self.shape.1];
        }
        Ok(matrix)
    }

    /// Divide a scalar from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let new_matrix = matrix.div_scalar(2);
    ///
    /// assert_eq!(new_matrix[0][0], 0);
    /// assert_eq!(new_matrix[0][1], 1);
    /// assert_eq!(new_matrix[0][2], 1);
    /// assert_eq!(new_matrix[1][0], 2);
    /// assert_eq!(new_matrix[1][1], 2);
    /// assert_eq!(new_matrix[1][2], 3);
    /// ```
    pub fn div_scalar(&self, scalar: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] / scalar;
        }
        matrix
    }
}
