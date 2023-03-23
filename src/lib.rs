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
//! ```rust
//! use matrix_operations::Matrix;
//!
//! // Create a matrix from a 2 dimensional vector
//! let mut m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
//!
//! // Create a matrix from a 1 dimensional vector and a shape
//! let m2 = Matrix::new(vec![1, 2, 3, 4, 5, 6], (3, 2)).unwrap();
//!
//! // Modify a matrix
//! m1[0][0] = 10;
//!
//! // Multiply two matrices
//! let m3 = m1.dot(&m2).unwrap();
//!
//! // Apply a function to each element of a matrix
//! let m4 = m3.apply(|x| x * 2);
//!
//! // Apply a function to each element of two matrices
//! let m5 = m3.apply_other(&m4, |x, y| x * y).unwrap();
//!
//! println!("{}", m5);
//! ```
//!
//! ## Features
//!
//! - Create a matrix
//! - Transpose a matrix
//! - Multiply two matrices
//! - Add two matrices
//! - Subtract two matrices
//! - Apply a function to each element of a matrix (like multiplying by a scalar, or adding a constant)
//! - Apply a function on each element of two matrices (like multiplying two matrices element by element)
//!

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub};

/// A matrix struct that can be used to perform matrix operations.
pub struct Matrix<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: (usize, usize),
}

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
impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.shape.1;
        let end = start + self.shape.1;
        &self.data[start..end]
    }
}

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
impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.shape.1;
        let end = start + self.shape.1;
        &mut self.data[start..end]
    }
}

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
impl<T> Display for Matrix<T> where T: Display {
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

    /// Apply a function to each element of the matrix
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
    /// let new_matrix = matrix.apply(|x| x * 2);
    ///
    /// assert_eq!(new_matrix[0][0], 2);
    /// assert_eq!(new_matrix[0][1], 4);
    /// assert_eq!(new_matrix[0][2], 6);
    /// assert_eq!(new_matrix[1][0], 8);
    /// assert_eq!(new_matrix[1][1], 10);
    /// assert_eq!(new_matrix[1][2], 12);
    /// ```
    pub fn apply(&self, f: fn(T) -> T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = f(self[i][j]);
            }
        }
        matrix
    }

    /// Apply a function to each element of two matrices and return a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::Matrix;
    ///
    /// let shape = (2, 3);
    ///
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape).unwrap();
    ///
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix2 = Matrix::new(data2, shape).unwrap();
    ///
    /// let new_matrix = matrix1.apply_other(&matrix2, |x, y| x + y).unwrap();
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
    /// let new_matrix = matrix1.apply_other(&matrix2, |x, y| x + y);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn apply_other(&self, other: &Matrix<T>, f: fn(T, T) -> T) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape != other.shape {
            return Err("Matrix shapes are not compatible to apply function".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = f(self[i][j], other[i][j]);
            }
        }
        Ok(matrix)
    }

    /// Transpose the matrix
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
    /// let new_matrix = matrix.transpose();
    ///
    /// assert_eq!(new_matrix[0][0], 1);
    /// assert_eq!(new_matrix[0][1], 4);
    /// assert_eq!(new_matrix[1][0], 2);
    /// assert_eq!(new_matrix[1][1], 5);
    /// assert_eq!(new_matrix[2][0], 3);
    /// assert_eq!(new_matrix[2][1], 6);
    /// assert_eq!(new_matrix.shape(), (3, 2));
    /// ```
    pub fn transpose(&self) -> Matrix<T> {
        let mut matrix = Matrix::default((self.shape.1, self.shape.0));
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[j][i] = self[i][j];
            }
        }
        matrix
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
        for i in 0..self.shape.0 {
            for j in 0..other.shape.1 {
                for k in 0..self.shape.1 {
                    matrix[i][j] += self[i][k] * other[k][j];
                }
            }
        }
        Ok(matrix)
    }
}

impl<T: Default + Copy + Add<Output = T>> Matrix<T> {

    /// Add two matrices together
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
    /// let new_matrix = matrix1.add(&matrix2).unwrap();
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
    /// use matrix_operations::Matrix;
    ///
    /// let shape1 = (2, 3);
    /// let shape2 = (3, 2);
    /// let data1 = vec![1, 2, 3, 4, 5, 6];
    /// let data2 = vec![1, 2, 3, 4, 5, 6];
    /// let matrix1 = Matrix::new(data1, shape1).unwrap();
    /// let matrix2 = Matrix::new(data2, shape2).unwrap();
    ///
    /// let new_matrix = matrix1.add(&matrix2);
    ///
    /// assert!(new_matrix.is_err());
    /// ```
    pub fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape != other.shape {
            return Err("Matrix shapes are not compatible for addition".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] + other.data[i];
        }
        Ok(matrix)
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
}
