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

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut, Range};

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

impl<T: Copy + Default> Index<Range<usize>> for Matrix<T> {
    type Output = [T];

    /// Allows the matrix get a range of rows
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (3, 2);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// assert_eq!(matrix[0..1], [1, 2]);
    /// assert_eq!(matrix[1..3], [3, 4, 5, 6]);
    /// ```
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (3, 2);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// let matrix2 = Matrix::from_slice(&matrix[0..2], (2, 2)).unwrap();
    ///
    /// assert_eq!(matrix2.as_slice(), [1, 2, 3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds
    ///
    /// ```should_panic
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (3, 2);
    /// let matrix = Matrix::new(data, shape).unwrap();
    ///
    /// assert_eq!(matrix[0..4], [1, 2, 3, 4, 5, 6]);
    /// ```
    fn index(&self, index: Range<usize>) -> &Self::Output {
        let start = index.start * self.shape.1;
        let end = index.end * self.shape.1;
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
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_column(0).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 4));
    /// assert_eq!(matrix.as_slice(), [0, 1, 2, 3, 0, 4, 5, 6]);
    ///
    /// matrix.add_column(matrix.shape().1).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 5));
    /// assert_eq!(matrix.as_slice(), [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_column(0).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (1, 1));
    /// assert_eq!(matrix.as_slice(), [0]);
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
    /// let mut matrix = Matrix::new(data, shape).unwrap();
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_column_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 4));
    /// assert_eq!(matrix.as_slice(), [10, 1, 2, 3, 10, 4, 5, 6]);
    ///
    /// matrix.add_column_initialized(matrix.shape().1, 20).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 5));
    /// assert_eq!(matrix.as_slice(), [10, 1, 2, 3, 20, 10, 4, 5, 6, 20]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_column_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (1, 1));
    /// assert_eq!(matrix.as_slice(), [10]);
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
    /// let mut matrix = Matrix::new(data, shape).unwrap();
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_column_from_vec(0, vec![10, 20]).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 4));
    /// assert_eq!(matrix.as_slice(), [10, 1, 2, 3, 20, 4, 5, 6]);
    ///
    /// matrix.add_column_from_vec(matrix.shape().1, vec![30, 40]).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 5));
    /// assert_eq!(matrix.as_slice(), [10, 1, 2, 3, 30, 20, 4, 5, 6, 40]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the column
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_column_from_vec(0, vec![10, 20]).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2, 1));
    /// assert_eq!(matrix.as_slice(), [10, 20]);
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
    /// let mut matrix = Matrix::new(data, shape).unwrap();
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

    /// Add a row to the matrix
    /// The row will be added to the index specified
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_row(1).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (3, 3));
    /// assert_eq!(matrix.as_slice(), [1, 2, 3, 0, 0, 0, 4, 5, 6]);
    ///
    /// matrix.add_row(matrix.shape().0).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (4, 3));
    /// assert_eq!(matrix.as_slice(), [1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_row(0).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (1, 1));
    /// assert_eq!(matrix.as_slice(), [0]);
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
    /// let mut matrix = Matrix::new(data, shape).unwrap();
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
    /// use matrix_operations::Matrix;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let shape = (2, 3);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_row_initialized(1, 10).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (3, 3));
    /// assert_eq!(matrix.as_slice(), [1, 2, 3, 10, 10, 10, 4, 5, 6]);
    ///
    /// matrix.add_row_initialized(matrix.shape().0, 20).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (4, 3));
    /// assert_eq!(matrix.as_slice(), [1, 2, 3, 10, 10, 10, 4, 5, 6, 20, 20, 20]);
    /// ```
    ///
    /// If the matrix is empty, a new matrix will be created with the row
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let data: Vec<u32> = Vec::new();
    /// let shape = (0, 0);
    /// let mut matrix = Matrix::new(data, shape).unwrap();
    ///
    /// matrix.add_row_initialized(0, 10).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (1, 1));
    /// assert_eq!(matrix.as_slice(), [10]);
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
    /// let mut matrix = Matrix::new(data, shape).unwrap();
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
}
