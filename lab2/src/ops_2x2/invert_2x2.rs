use nalgebra::{DMatrix, DMatrixView};

use crate::{ops_2x2::{binet, strassen}};

fn invert_2x2(a: DMatrixView<f64>, multiply: fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>)) -> (usize, DMatrix<f64>) {
    todo!()
}

pub fn invert_binet(a: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    invert_2x2(a.as_view(), binet::calculate)
}

pub fn invert_strassen(a: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    invert_2x2(a.as_view(), strassen::calculate)
}

