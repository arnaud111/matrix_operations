use std::ops::{AddAssign, Mul, MulAssign};
use crate::{dot_matrices, Matrix, mul_matrix_with_scalar};

impl<T: Copy + Default + MulAssign> Matrix<T> {

    /// Allows the matrix to be multiplied to a scalar with operator `*=`
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// matrix1.mul_scalar(2);
    ///
    /// assert_eq!(matrix1, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    pub fn mul_scalar(&mut self, scalar: T) {
        for i in 0..self.data.len() {
            self.data[i] *= scalar;
        }
    }
}

impl<T: Copy + Default + Mul<Output = T> + AddAssign> Matrix<T> {

    /// Allows the matrix to be multiplied to another matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2],
    ///                       [3, 4],
    ///                       [5, 6]];
    ///
    /// matrix1.dot_matrix(&matrix2);
    ///
    /// assert_eq!(matrix1, matrix![[22, 28], [49, 64]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the matrices can't be multiplied
    ///
    /// ```should_panic
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// // Panics
    /// matrix1.dot_matrix(&matrix2);
    /// ```
    pub fn dot_matrix(&mut self, other: &Matrix<T>) {
        *self = dot_matrices(self, other).unwrap();
    }
}

impl<T: Copy + Default + MulAssign> MulAssign<T> for Matrix<T> {

    /// Allows the matrix to be multiplied to a scalar with operator `*=`
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// matrix1 *= 2;
    ///
    /// assert_eq!(matrix1, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul_assign(&mut self, scalar: T) {
        self.mul_scalar(scalar);
    }
}

impl<T: Copy + Default + Mul<Output = T> + AddAssign> MulAssign for Matrix<T> {

    /// Allows the matrix to be multiplied to another matrix with operator `*=`
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2],
    ///                       [3, 4],
    ///                       [5, 6]];
    ///
    /// matrix1 *= matrix2;
    ///
    /// assert_eq!(matrix1, matrix![[22, 28], [49, 64]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the matrices can't be multiplied
    ///
    /// ```should_panic
    /// use matrix_operations::matrix;
    ///
    /// let mut matrix1 = matrix![[1, 2, 3],
    ///                           [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// // Panics
    /// matrix1 *= matrix2;
    /// ```
    fn mul_assign(&mut self, matrix: Self) {
        self.dot_matrix(&matrix);
    }
}

impl<T: Copy + Default + Mul<Output = T> + AddAssign> Mul for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be multiplied to another matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2],
    ///                       [3, 4],
    ///                       [5, 6]];
    ///
    /// let matrix3 = matrix1 * matrix2;
    ///
    /// assert_eq!(matrix3, matrix![[22, 28], [49, 64]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the matrices can't be multiplied
    ///
    /// ```should_panic
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// // Panics
    /// let matrix3 = matrix1 * matrix2;
    /// ```
    fn mul(self, other: Matrix<T>) -> Self::Output {
        dot_matrices(&self, &other).unwrap()
    }
}

impl<T: Copy + Default + Mul<Output = T>> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::matrix;
    ///
    /// let matrix1 = matrix![[1, 2, 3],
    ///                       [4, 5, 6]];
    ///
    /// let matrix2 = matrix1 * 2;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, scalar: T) -> Self::Output {
        mul_matrix_with_scalar(&self, scalar)
    }
}

impl Mul<Matrix<u8>> for u8 {
    type Output = Matrix<u8>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u8> = matrix![[1, 2, 3],
    ///                                   [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<u8>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<i8>> for i8 {
    type Output = Matrix<i8>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i8> = matrix![[1, 2, 3],
    ///                                   [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<i8>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<u16>> for u16 {
    type Output = Matrix<u16>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u16> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<u16>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<i16>> for i16 {
    type Output = Matrix<i16>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i16> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<i16>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<u32>> for u32 {
    type Output = Matrix<u32>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u32> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<u32>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<i32>> for i32 {
    type Output = Matrix<i32>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i32> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<i32>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<u64>> for u64 {
    type Output = Matrix<u64>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u64> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<u64>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<i64>> for i64 {
    type Output = Matrix<i64>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i64> = matrix![[1, 2, 3],
    ///                                    [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<i64>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<u128>> for u128 {
    type Output = Matrix<u128>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<u128> = matrix![[1, 2, 3],
    ///                                     [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<u128>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<i128>> for i128 {
    type Output = Matrix<i128>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<i128> = matrix![[1, 2, 3],
    ///                                     [4, 5, 6]];
    ///
    /// let matrix2 = 2 * matrix1;
    ///
    /// assert_eq!(matrix2, matrix![[2, 4, 6], [8, 10, 12]]);
    /// ```
    fn mul(self, matrix: Matrix<i128>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<f32> = matrix![[1.0, 2.0, 3.0],
    ///                                    [4.0, 5.0, 6.0]];
    ///
    /// let matrix2 = 2.0 * matrix1;
    ///
    /// assert_eq!(matrix2[0], [2.0, 4.0, 6.0]);
    /// assert_eq!(matrix2[1], [8.0, 10.0, 12.0]);
    /// ```
    fn mul(self, matrix: Matrix<f32>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}

impl Mul<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    /// Allows the matrix to be multiplied to a scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_operations::{matrix, Matrix};
    ///
    /// let matrix1: Matrix<f64> = matrix![[1.0, 2.0, 3.0],
    ///                                    [4.0, 5.0, 6.0]];
    ///
    /// let matrix2 = 2.0 * matrix1;
    ///
    /// assert_eq!(matrix2[0], [2.0, 4.0, 6.0]);
    /// assert_eq!(matrix2[1], [8.0, 10.0, 12.0]);
    /// ```
    fn mul(self, matrix: Matrix<f64>) -> Self::Output {
        mul_matrix_with_scalar(&matrix, self)
    }
}
