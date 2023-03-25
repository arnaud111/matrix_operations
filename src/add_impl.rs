use std::ops::Add;
use crate::{add_matrices, add_matrix_with_1col_matrix, add_matrix_with_1row_matrix, add_matrix_with_scalar, Matrix};

impl<T: Copy + Default + Add<Output = T>> Add for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be added to another matrix
    ///
    /// Works too for adding a matrix to an other matrix with 1 row or 1 column
    ///
    /// # Examples
    ///
    /// Same shape :
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
    /// let matrix3 = matrix1 + matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    ///
    /// Matrix with 1 row :
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2, 3]];
    ///
    /// let matrix3 = matrix1 + matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[2, 4, 6], [5, 7, 9]]);
    /// ```
    ///
    /// Matrix with 1 column :
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1],
    ///                       [4]];
    ///
    /// let matrix3 = matrix1 + matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[2, 3, 4], [8, 9, 10]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes and no one of them has 1 row or 1 column
    ///
    /// ```should_panic
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2],
    ///                       [3, 4],
    ///                       [5, 6]];
    ///
    /// // Panics
    /// let matrix3 = matrix1 + matrix2;
    /// ```
    fn add(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape != other.shape {
            return if other.shape.0 == 1 {
                add_matrix_with_1row_matrix(&self, &other).unwrap()
            } else if other.shape.1 == 1 {
                add_matrix_with_1col_matrix(&self, &other).unwrap()
            } else if self.shape.0 == 1 {
                add_matrix_with_1row_matrix(&other, &self).unwrap()
            } else {
                add_matrix_with_1col_matrix(&other, &self).unwrap()
            }
        }
        add_matrices(&self, &other).unwrap()
    }
}

impl<T: Copy + Default + Add<Output = T>> Add<T> for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix1 + 1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, scalar: T) -> Matrix<T> {
        add_matrix_with_scalar(&self, scalar)
    }
}

impl Add<Matrix<u16>> for u16 {
    type Output = Matrix<u16>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u16> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<u16>) -> Matrix<u16> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<u32>> for u32 {
    type Output = Matrix<u32>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u32> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<u32>) -> Matrix<u32> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<u64>> for u64 {
    type Output = Matrix<u64>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u64> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<u64>) -> Matrix<u64> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<u128>> for u128 {
    type Output = Matrix<u128>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u128> = matrix![[1, 2, 3],
    ///                                     [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<u128>) -> Matrix<u128> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<i8>> for i8 {
    type Output = Matrix<i8>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i8> = matrix![[1, 2, 3],
    ///                                   [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<i8>) -> Matrix<i8> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<i16>> for i16 {
    type Output = Matrix<i16>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i16> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<i16>) -> Matrix<i16> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<i32>> for i32 {
    type Output = Matrix<i32>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i32> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<i32>) -> Matrix<i32> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<i64>> for i64 {
    type Output = Matrix<i64>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i64> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<i64>) -> Matrix<i64> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<i128>> for i128 {
    type Output = Matrix<i128>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i128> = matrix![[1, 2, 3],
    ///                                     [4, 5, 6]];
    ///
    /// let matrix2 = 1 + matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    fn add(self, matrix: Matrix<i128>) -> Matrix<i128> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<f32> = matrix![[1.0, 2.0, 3.0],
    ///                                    [4.0, 5.0, 6.0]];
    ///
    /// let matrix2 = 1.0 + matrix1;
    ///
    /// assert_eq!(matrix2[0], [2.0, 3.0, 4.0]);
    /// assert_eq!(matrix2[1], [5.0, 6.0, 7.0]);
    /// ```
    fn add(self, matrix: Matrix<f32>) -> Matrix<f32> {
        add_matrix_with_scalar(&matrix, self)
    }
}

impl Add<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<f64> = matrix![[1.0, 2.0, 3.0],
    ///                                    [4.0, 5.0, 6.0]];
    ///
    /// let matrix2 = 1.0 + matrix1;
    ///
    /// assert_eq!(matrix2[0], [2.0, 3.0, 4.0]);
    /// assert_eq!(matrix2[1], [5.0, 6.0, 7.0]);
    /// ```
    fn add(self, matrix: Matrix<f64>) -> Matrix<f64> {
        add_matrix_with_scalar(&matrix, self)
    }
}

