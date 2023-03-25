use std::ops::{Div, DivAssign};
use crate::{apply_to_matrix_with_param, div_matrix_with_scalar, Matrix};

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

impl<T: Copy + Default + DivAssign> Matrix<T> {

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// matrix1.div_scalar(2);
    ///
    /// assert_eq!(matrix1, matrix![[0, 1, 1], [2, 2, 3]]);
    /// ```
    pub fn div_scalar(&mut self, scalar: T) {
        for i in 0..self.data.len() {
            self.data[i] /= scalar;
        }
    }
}

impl<T: Copy + Default + DivAssign> DivAssign<T> for Matrix<T> {

    /// Allows the matrix to be divided to a scalar with operator `/=`
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// matrix1 /= 2;
    ///
    /// assert_eq!(matrix1, matrix![[0, 1, 1], [2, 2, 3]]);
    /// ```
    fn div_assign(&mut self, scalar: T) {
        self.div_scalar(scalar);
    }
}

impl Div<Matrix<u8>> for u8 {
    type Output = Matrix<u8>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u8> = matrix![[1, 2, 3],
    ///                                   [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<u8>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<i8>> for i8 {
    type Output = Matrix<i8>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i8> = matrix![[1, 2, 3],
    ///                                   [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<i8>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<u16>> for u16 {
    type Output = Matrix<u16>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u16> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<u16>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<i16>> for i16 {
    type Output = Matrix<i16>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i16> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<i16>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<u32>> for u32 {
    type Output = Matrix<u32>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u32> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<u32>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<i32>> for i32 {
    type Output = Matrix<i32>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i32> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<i32>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<u64>> for u64 {
    type Output = Matrix<u64>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u64> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<u64>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<i64>> for i64 {
    type Output = Matrix<i64>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i64> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<i64>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<u128>> for u128 {
    type Output = Matrix<u128>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u128> = matrix![[1, 2, 3],
    ///                                     [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<u128>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<i128>> for i128 {
    type Output = Matrix<i128>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i128> = matrix![[1, 2, 3],
    ///                                     [4, 5, 6]];
    ///
    /// let matrix2 = 2 / matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 1, 0], [0, 0, 0]]);
    /// ```
    fn div(self, matrix: Matrix<i128>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<f32> = matrix![[1.0, 2.0, 3.0],
    ///                                    [4.0, 5.0, 6.0]];
    ///
    /// let matrix2 = 2.0 / matrix1;
    ///
    /// assert_eq!(matrix2[0], [2.0, 1.0, 0.6666667]);
    /// assert_eq!(matrix2[1], [0.5, 0.4, 0.33333334]);
    /// ```
    fn div(self, matrix: Matrix<f32>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}

impl Div<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    /// Allows the matrix to be divided to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<f64> = matrix![[1.0, 2.0, 3.0],
    ///                                    [4.0, 5.0, 6.0]];
    ///
    /// let matrix2 = 2.0 / matrix1;
    ///
    /// assert_eq!(matrix2[0], [2.0, 1.0, 0.6666666666666666]);
    /// assert_eq!(matrix2[1], [0.5, 0.4, 0.3333333333333333]);
    /// ```
    fn div(self, matrix: Matrix<f64>) -> Self::Output {
        apply_to_matrix_with_param(&matrix, |x, y| y / x, self)
    }
}
