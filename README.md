# Matrix Operations

[![Crates.io](https://img.shields.io/crates/v/matrix_operations.svg)](https://crates.io/crates/matrix_operations)
[![Docs.rs](https://docs.rs/matrix_operations/badge.svg)](https://docs.rs/matrix_operations)

Matrix_Operations is a Rust crate for performing various matrix operations. It provides a set of functions for performing common matrix operations.

## Installation

Add the following to your Cargo.toml file:

```toml
[dependencies]
matrix_operations = "0.1.0"
```

## Usage

This crate provides a wide range of operations that can be performed on matrices. Here are some examples of common operations:

```rust
use matrix_operations::matrix;
use matrix_operations::operations::transpose_matrix;

let matrix1 = matrix![[1, 2, 3],
[4, 5, 6]];

let matrix2 = matrix![[7, 8, 9],
[10, 11, 12]];

let mut matrix3 = matrix1.clone() + matrix2.clone() * 2;
assert_eq!(matrix3, matrix![[15, 18, 21], [24, 27, 30]]);

matrix3 -= 1;
assert_eq!(matrix3, matrix![[14, 17, 20], [23, 26, 29]]);

matrix3 /= 2;
assert_eq!(matrix3, matrix![[7, 8, 10], [11, 13, 14]]);

matrix3 -= matrix1;
assert_eq!(matrix3, matrix![[6, 6, 7], [7, 8, 8]]);

matrix3 = transpose_matrix(&matrix3);
assert_eq!(matrix3, matrix![[6, 7], [6, 8], [7, 8]]);

matrix3 *= matrix2;
assert_eq!(matrix3, matrix![[112, 125, 138], [122, 136, 150], [129, 144, 159]]);
```

This crate also provides functionality to load and save matrices to a file:

```rust
use matrix_operations::csv::{load_matrix_from_csv, write_matrix_to_csv};
use matrix_operations::matrix;

let matrix1 = matrix![[1, 2, 3],
                      [4, 5, 6]];

write_matrix_to_csv(&matrix1, "resources/matrix.csv", ",").unwrap();

let matrix2 = load_matrix_from_csv("resources/matrix.csv", ",").unwrap();

assert_eq!(matrix1, matrix2);
```

## Features
- Create a matrix
- Transpose a matrix
- Multiply / Add / Subtract two matrices
- Multiply / Divide / Add / Subtract a matrix by a scalar
- Add / Subtract each matrix rows / columns by a distinct value
- Apply a function to each element of a matrix (like multiplying by a scalar, or adding a constant)
- Apply a function on each element of two matrices (like multiplying two matrices element by element)
- Apply a function on each row or column of a matrix
- Get a matrix as a slice / vector / 2D vector
- Load / Save a matrix in a file
- Add / Remove rows / columns
- Concatenate matrices
