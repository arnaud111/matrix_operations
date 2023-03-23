use std::error::Error;
use std::ops::{Add, AddAssign, Mul, Sub};
use crate::Matrix;

impl<T: Default + Copy> Matrix<T> {

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

impl<T: Default + Copy + Mul + AddAssign<<T as Mul>::Output>> Matrix<T> {

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

    pub fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape != other.shape {
            return Err("Matrix shapes are not compatible for addition".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] + other[i][j];
            }
        }
        Ok(matrix)
    }
}

impl<T: Default + Copy + Sub<Output = T>> Matrix<T> {

    pub fn sub(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape != other.shape {
            return Err("Matrix shapes are not compatible for addition".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] - other[i][j];
            }
        }
        Ok(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4], vec![2, 5], vec![3, 6]]).unwrap();
        let m1 = m1.transpose();
        assert_eq!(m1.shape, m2.shape);
        for i in 0..m1.shape.0 {
            for j in 0..m1.shape.1 {
                assert_eq!(m1[i][j], m2[i][j]);
            }
        }
    }

    #[test]
    fn test_dot() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4], vec![2, 5], vec![3, 6]]).unwrap();
        let m3 = Matrix::from_2d_vec(vec![vec![14, 32], vec![32, 77]]).unwrap();
        let m4 = m1.dot(&m2).unwrap();
        assert_eq!(m3.shape, m4.shape);
        for i in 0..m3.shape.0 {
            for j in 0..m3.shape.1 {
                assert_eq!(m3[i][j], m4[i][j]);
            }
        }
    }

    #[test]
    fn test_dot_error() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4], vec![2, 5]]).unwrap();
        let m3 = m1.dot(&m2);
        assert!(m3.is_err());
    }

    #[test]
    fn test_add() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4, 7], vec![2, 5, 8]]).unwrap();
        let m3 = Matrix::from_2d_vec(vec![vec![2, 6, 10], vec![6, 10, 14]]).unwrap();
        let m4 = m1.add(&m2).unwrap();
        assert_eq!(m3.shape, m4.shape);
        for i in 0..m3.shape.0 {
            for j in 0..m3.shape.1 {
                assert_eq!(m3[i][j], m4[i][j]);
            }
        }
    }

    #[test]
    fn test_add_error() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4], vec![2, 5]]).unwrap();
        let m3 = m1.add(&m2);
        assert!(m3.is_err());
    }

    #[test]
    fn test_sub() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4, 7], vec![2, 5, 8]]).unwrap();
        let m3 = Matrix::from_2d_vec(vec![vec![0, -2, -4], vec![2, 0, -2]]).unwrap();
        let m4 = m1.sub(&m2).unwrap();
        assert_eq!(m3.shape, m4.shape);
        for i in 0..m3.shape.0 {
            for j in 0..m3.shape.1 {
                assert_eq!(m3[i][j], m4[i][j]);
            }
        }
    }

    #[test]
    fn test_sub_error() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 4], vec![2, 5]]).unwrap();
        let m3 = m1.sub(&m2);
        assert!(m3.is_err());
    }
}