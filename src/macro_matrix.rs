
/// This macro is used to create a matrix from a 2D array of numbers.
///
/// # Examples
///
/// ```
/// use matrix_operations::matrix;
///
/// let m = matrix![[1, 2, 3],
///                 [4, 5, 6],
///                 [7, 8, 9]];
///
/// assert_eq!(m[0], vec![1, 2, 3]);
/// assert_eq!(m[1], vec![4, 5, 6]);
/// assert_eq!(m[2], vec![7, 8, 9]);
/// ```
///
/// # Panics
///
/// This macro will panic if the matrix is not rectangular.
///
/// ```should_panic
/// use matrix_operations::matrix;
///
/// let m = matrix![[1, 2, 3],
///                 [4, 5, 6],
///                 [7, 8]];
/// ```
#[macro_export]
macro_rules! matrix {
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {
        $crate::Matrix::from_2d_vec(vec![$(vec![$($x),*]),+]).unwrap()
    };
    () => {
        $crate::Matrix::new(vec![], (0, 0)).unwrap()
    };
}