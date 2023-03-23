use std::error::Error;

pub struct Matrix<T, const N: usize> {
    data: [T; N],
    shape: (usize, usize),
}

impl<T, const N: usize> Matrix<T, N> {
    pub fn new(data: [T; N], shape: (usize, usize)) -> Result<Matrix<T, N>, Box<dyn Error>> {
        if (shape.0 * shape.1) != N {
            return Err("Invalid shape".into());
        }
        Ok(Matrix {
            data,
            shape
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let matrix = Matrix::new(data, (3, 3)).unwrap();
        assert_eq!(matrix.shape, (3, 3));
    }

    #[test]
    fn test_matrix_invalid_shape() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let matrix = Matrix::new(data, (3, 4));
        assert!(matrix.is_err());
    }
}