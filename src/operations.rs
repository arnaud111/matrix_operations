use crate::Matrix;

impl<T: Default + std::marker::Copy> Matrix<T> {

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose() {
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
}