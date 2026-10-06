use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};

use crate::util::split_view;

pub fn mul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let mut result = DMatrix::zeros(n, n);
    add_multiplied(&mut result.view_range_mut(.., ..), &a.view_range(.., ..), &b.view_range(.., ..));
    result
}

fn add_multiplied(result: &mut DMatrixViewMut<f64>, a: &DMatrixView<f64>, b: &DMatrixView<f64>) {
    let n = a.ncols();
    if a.nrows() == 1 {
        result[(0, 0)] += a[(0, 0)] * b[(0, 0)];
    } else {
        let split = n / 2;
        let (a11, a12, a21, a22) = split_view(a);
        let (b11, b12, b21, b22) = split_view(b);
        add_multiplied(&mut result.view_range_mut(..split, ..split), &a11, &b11);
        add_multiplied(&mut result.view_range_mut(..split, ..split), &a12, &b21);
        add_multiplied(&mut result.view_range_mut(..split, split..), &a11, &b12);
        add_multiplied(&mut result.view_range_mut(..split, split..), &a12, &b22);
        add_multiplied(&mut result.view_range_mut(split.., ..split), &a21, &b11);
        add_multiplied(&mut result.view_range_mut(split.., ..split), &a22, &b21);
        add_multiplied(&mut result.view_range_mut(split.., split..), &a21, &b12);
        add_multiplied(&mut result.view_range_mut(split.., split..), &a22, &b22);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn mul1x1() {
        assert_eq!(
            DMatrix::from_row_slice(1, 1, &[6.]),
            mul(&DMatrix::from_row_slice(1, 1, &[2.]), &DMatrix::from_row_slice(1, 1, &[3.]))
        );
    }
    
    #[test]
    fn mul2x2() {
        assert_eq!(
            DMatrix::from_row_slice(2, 2, &[
                19., 22.,
                43., 50.,
            ]),
            mul(&DMatrix::from_row_slice(2, 2, &[
                1., 2.,
                3., 4.,
            ]), &DMatrix::from_row_slice(2, 2, &[
                5., 6.,
                7., 8.,
            ]))
        );
    }
}
