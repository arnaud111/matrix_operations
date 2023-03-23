use std::error::Error;
use std::ops::{Add, AddAssign, Div, Mul, Sub};
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

impl<T: Default + Copy + Mul<Output = T> + AddAssign<<T as Mul>::Output>> Matrix<T> {

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

    pub fn multiply_by_value(&self, value: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] * value;
            }
        }
        matrix
    }

    pub fn multiply_one_by_one(&self, other: &Matrix<T>) -> Result<Matrix<T>, Box<dyn Error>> {
        if self.shape != other.shape {
            return Err("Matrix shapes are not compatible for element-wise multiplication".into());
        }
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] * other[i][j];
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

    pub fn add_value(&self, value: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] + value;
            }
        }
        matrix
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

    pub fn sub_value(&self, value: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] - value;
            }
        }
        matrix
    }

    pub fn value_sub(&self, value: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = value - self[i][j];
            }
        }
        matrix
    }
}

impl<T: Default + Copy + Div<Output = T>> Matrix<T> {

    pub fn div_by_value(&self, value: T) -> Matrix<T> {
        let mut matrix = Matrix::default(self.shape);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                matrix[i][j] = self[i][j] / value;
            }
        }
        matrix
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

    #[test]
    fn test_multiply_by_value() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![2, 4, 6], vec![8, 10, 12]]).unwrap();
        let m3 = m1.multiply_by_value(2);
        assert_eq!(m2.shape, m3.shape);
        for i in 0..m2.shape.0 {
            for j in 0..m2.shape.1 {
                assert_eq!(m2[i][j], m3[i][j]);
            }
        }
    }

    #[test]
    fn test_add_value() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![2, 3, 4], vec![5, 6, 7]]).unwrap();
        let m3 = m1.add_value(1);
        assert_eq!(m2.shape, m3.shape);
        for i in 0..m2.shape.0 {
            for j in 0..m2.shape.1 {
                assert_eq!(m2[i][j], m3[i][j]);
            }
        }
    }

    #[test]
    fn test_sub_value() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![0, 1, 2], vec![3, 4, 5]]).unwrap();
        let m3 = m1.sub_value(1);
        assert_eq!(m2.shape, m3.shape);
        for i in 0..m2.shape.0 {
            for j in 0..m2.shape.1 {
                assert_eq!(m2[i][j], m3[i][j]);
            }
        }
    }

    #[test]
    fn test_value_sub() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![0, -1, -2], vec![-3, -4, -5]]).unwrap();
        let m3 = m1.value_sub(1);
        assert_eq!(m2.shape, m3.shape);
        for i in 0..m2.shape.0 {
            for j in 0..m2.shape.1 {
                assert_eq!(m2[i][j], m3[i][j]);
            }
        }
    }

    #[test]
    fn test_div_by_value() {
        let m1 = Matrix::from_2d_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]]).unwrap();
        let m3 = m1.div_by_value(2.0);
        assert_eq!(m2.shape, m3.shape);
        for i in 0..m2.shape.0 {
            for j in 0..m2.shape.1 {
                assert_eq!(m2[i][j], m3[i][j]);
            }
        }
    }

    #[test]
    fn test_multiply_one_by_one() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m3 = Matrix::from_2d_vec(vec![vec![1, 4, 9], vec![16, 25, 36]]).unwrap();
        let m4 = m1.multiply_one_by_one(&m2).unwrap();
        assert_eq!(m3.shape, m4.shape);
        for i in 0..m3.shape.0 {
            for j in 0..m3.shape.1 {
                assert_eq!(m3[i][j], m4[i][j]);
            }
        }
    }

    #[test]
    fn test_multiply_one_by_one_error() {
        let m1 = Matrix::from_2d_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let m2 = Matrix::from_2d_vec(vec![vec![1, 2], vec![4, 5]]).unwrap();
        let m3 = m1.multiply_one_by_one(&m2);
        assert!(m3.is_err());
    }
}