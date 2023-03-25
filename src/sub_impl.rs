use std::ops::Sub;
use crate::{apply_to_matrix_with_param, Matrix, sub_matrices, sub_matrix_with_1col_matrix, sub_matrix_with_1row_matrix, sub_matrix_with_scalar};

impl<T: Copy + Default + Sub<Output= T>> Sub for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be subtracted to another matrix
    ///
    /// Works too for subtracting a matrix to an other matrix with 1 row or 1 column
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
    /// let matrix3 = matrix1 - matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[0, 0, 0], [0, 0, 0]]);
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
    /// let matrix3 = matrix1 - matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[0, 0, 0], [3, 3, 3]]);
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
    ///                       [2]];
    ///
    /// let matrix3 = matrix1 - matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[0, 1, 2], [2, 3, 4]]);
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
    /// let matrix3 = matrix1 - matrix2;
    /// ```
    fn sub(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            return if other.shape.0 == 1 {
                sub_matrix_with_1row_matrix(&self, &other).unwrap()
            } else if other.shape.1 == 1 {
                sub_matrix_with_1col_matrix(&self, &other).unwrap()
            } else if self.shape.0 == 1 {
                sub_matrix_with_1row_matrix(&other, &self).unwrap()
            } else {
                sub_matrix_with_1col_matrix(&other, &self).unwrap()
            }
        }
        sub_matrices(&self, &other).unwrap()
    }
}

impl<T: Copy + Default + Sub<Output = T>> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be subtracted to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix1 - 1;
    ///
    /// assert_eq!(matrix2, matrix![[0, 1, 2], [3, 4, 5]]);
    /// ```
    fn sub(self, scalar: T) -> Self::Output {
        sub_matrix_with_scalar(&self, scalar)
    }
}

impl Sub<Matrix<u16>> for u16 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<u16>) -> Matrix<u16> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<i16>> for i16 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<i16>) -> Matrix<i16> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<u32>> for u32 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<u32>) -> Matrix<u32> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<i32>> for i32 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<i32>) -> Matrix<i32> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<u64>> for u64 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<u64>) -> Matrix<u64> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<i64>> for i64 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<i64>) -> Matrix<i64> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<f32>> for f32 {
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
    /// let matrix2 = 7.0 - matrix1;
    ///
    /// assert_eq!(matrix2[0], [6.0, 5.0, 4.0]);
    /// assert_eq!(matrix2[1], [3.0, 2.0, 1.0]);
    /// ```
    fn sub(self, matrix: Matrix<f32>) -> Matrix<f32> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<f64>> for f64 {
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
    /// let matrix2 = 7.0 - matrix1;
    ///
    /// assert_eq!(matrix2[0], [6.0, 5.0, 4.0]);
    /// assert_eq!(matrix2[1], [3.0, 2.0, 1.0]);
    /// ```
    fn sub(self, matrix: Matrix<f64>) -> Matrix<f64> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<u128>> for u128 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<u128>) -> Matrix<u128> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<i128>> for i128 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<i128>) -> Matrix<i128> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<usize>> for usize {
    type Output = Matrix<usize>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<usize> = matrix![[1, 2, 3],
    ///                                      [4, 5, 6]];
    ///
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<usize>) -> Matrix<usize> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<isize>> for isize {
    type Output = Matrix<isize>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<isize> = matrix![[1, 2, 3],
    ///                                      [4, 5, 6]];
    ///
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<isize>) -> Matrix<isize> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<u8>> for u8 {
    type Output = Matrix<u8>;

    /// Allows the matrix to be added to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u8> = matrix![[1, 2, 3],
    ///                                   [4, 5, 6]];
    ///
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<u8>) -> Matrix<u8> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

impl Sub<Matrix<i8>> for i8 {
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
    /// let matrix2 = 7 - matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[6, 5, 4], [3, 2, 1]]);
    /// ```
    fn sub(self, matrix: Matrix<i8>) -> Matrix<i8> {
        apply_to_matrix_with_param(&matrix, |x, y| y - x, self)
    }
}

