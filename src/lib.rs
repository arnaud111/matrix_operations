use std::error::Error;

pub struct Matrix<T> {
    data: Vec<T>,
    shape: (usize, usize),
}

impl<T> std::ops::Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.shape.1;
        let end = start + self.shape.1;
        &self.data[start..end]
    }
}

impl<T: Default + std::marker::Copy> Matrix<T> {

    pub fn new(data: Vec<T>, shape: (usize, usize)) -> Result<Matrix<T>, Box<dyn Error>> {
        if data.len() != shape.0 * shape.1 {
            return Err("Data length does not match shape".into());
        }
        Ok(Matrix { data, shape })
    }

    pub fn default(shape: (usize, usize)) -> Matrix<T> {
        Matrix {
            data: vec![T::default(); shape.0 * shape.1],
            shape,
        }
    }

    pub fn from_slice(data: &[T], shape: (usize, usize)) -> Result<Matrix<T>, Box<dyn Error>> {
        if data.len() != shape.0 * shape.1 {
            return Err("Data length does not match shape".into());
        }
        Ok(Matrix {
            data: data.to_vec(),
            shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (2, 3);
        let matrix = Matrix::new(data, shape).unwrap();
        assert_eq!(matrix.data, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(matrix.shape, (2, 3));
    }

    #[test]
    fn test_new_error() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (3, 3);
        let matrix = Matrix::new(data, shape);
        assert!(matrix.is_err());
    }

    #[test]
    fn test_default() {
        let shape = (2, 3);
        let matrix: Matrix<u32> = Matrix::default(shape);
        assert_eq!(matrix.data, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(matrix.shape, (2, 3));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (2, 3);
        let matrix = Matrix::from_slice(&data, shape).unwrap();
        assert_eq!(matrix.data, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(matrix.shape, (2, 3));
    }

    #[test]
    fn test_from_slice_error() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (3, 3);
        let matrix = Matrix::from_slice(&data, shape);
        assert!(matrix.is_err());
    }

    #[test]
    fn test_index() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (2, 3);
        let matrix = Matrix::new(data, shape).unwrap();
        assert_eq!(matrix[0][0], 1);
        assert_eq!(matrix[0][1], 2);
        assert_eq!(matrix[0][2], 3);
        assert_eq!(matrix[1][0], 4);
        assert_eq!(matrix[1][1], 5);
        assert_eq!(matrix[1][2], 6);
    }
}