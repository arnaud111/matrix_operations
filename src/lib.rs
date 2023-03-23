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

impl<T> std::ops::IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.shape.1;
        let end = start + self.shape.1;
        &mut self.data[start..end]
    }
}

impl<T> std::fmt::Display for Matrix<T>
    where
        T: std::fmt::Display,
    {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    #[test]
    fn test_index_mut() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (2, 3);
        let mut matrix = Matrix::new(data, shape).unwrap();
        matrix[0][0] -= 7;
        matrix[0][1] += 8;
        matrix[0][2] /= 9;
        matrix[1][0] *= 10;
        matrix[1][1] %= 11;
        matrix[1][2] = 12;
        assert_eq!(matrix[0][0], 1 - 7);
        assert_eq!(matrix[0][1], 2 + 8);
        assert_eq!(matrix[0][2], 3 / 9);
        assert_eq!(matrix[1][0], 4 * 10);
        assert_eq!(matrix[1][1], 5 % 11);
        assert_eq!(matrix[1][2], 12);
    }

    #[test]
    fn test_display() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = (2, 3);
        let matrix = Matrix::new(data, shape).unwrap();
        assert_eq!(format!("{}", matrix), "1 2 3 \n4 5 6 \n");
    }
}