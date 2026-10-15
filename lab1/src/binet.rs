use nalgebra::{DMatrix, DMatrixView, dmatrix};

use crate::util::{dynamic_peeling, split_view};

fn calculate(a: DMatrixView<f64>, b: DMatrixView<f64>) -> (usize, DMatrix<f64>) {
    let n = a.nrows();
    
    if n == 1 {
        return (1, dmatrix![a[(0, 0)] * b[(0, 0)]]);
    }
    
    if n % 2 == 1 {
        return dynamic_peeling(&a, &b, calculate);
    }
    
    let (a11, a12, a21, a22) = split_view(&a);
    let (b11, b12, b21, b22) = split_view(&b);
    
    let (ops1, p1) = calculate(a11, b11);
    let (ops2, p2) = calculate(a12, b21);
    let (ops3, p3) = calculate(a11, b12);
    let (ops4, p4) = calculate(a12, b22);
    let (ops5, p5) = calculate(a21, b11);
    let (ops6, p6) = calculate(a22, b21);
    let (ops7, p7) = calculate(a21, b12);
    let (ops8, p8) = calculate(a22, b22);

    let mut result = DMatrix::zeros(n, n);
    let split_at = n / 2;
    result.view_range_mut(..split_at, ..split_at).copy_from(&(p1 + p2));
    result.view_range_mut(..split_at, split_at..).copy_from(&(p3 + p4));
    result.view_range_mut(split_at.., ..split_at).copy_from(&(p5 + p6));
    result.view_range_mut(split_at.., split_at..).copy_from(&(p7 + p8));
    let ops = ops1 + ops2 + ops3 + ops4 + ops5 + ops6 + ops7 + ops8 + 4 * split_at * split_at;
    (ops, result)
}

pub fn mul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    calculate(a.view_range(.., ..), b.view_range(.., ..))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn mul1x1() {
        assert_eq!((1, dmatrix![6.]), mul(&dmatrix![2.], &dmatrix![3.]));
    }
    
    #[test]
    fn mul2x2() {
        assert_eq!(
            (12, dmatrix![
                19., 22.;
                43., 50.;
            ]),
            mul(&dmatrix![
                1., 2.;
                3., 4.;
            ], &dmatrix![
                5., 6.;
                7., 8.;
            ])
        );
    }
}
