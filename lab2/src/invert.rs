use nalgebra::{DMatrix, DMatrixView, dmatrix};

use crate::binet;
use crate::laderman;
use crate::strassen;
use crate::util::{invert_peeling, split_view};

pub(super) fn invert(a: DMatrixView<f64>, multiply: fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>)) -> (usize, DMatrix<f64>) {
    let (n, _) = a.shape();

    if n == 1 {
        return (1, dmatrix![1. / a[(0, 0)]]);
    }

    if n % 2 != 0 { return invert_peeling(&a, invert, multiply) }

    let split_at = n / 2;
    let (a11, a12, 
         a21, a22) = split_view(&a);

    let (ops1, a11_inv) = invert(a11, multiply);
    let (ops2, s22) = {
        let (ops1, mid1) = multiply(a21, a11_inv.as_view());
        let (ops2, mid2) = multiply(mid1.as_view(), a12);
        let s22 = a22 - mid2;
        let ops3 = split_at * split_at;
        (ops1 + ops2 + ops3, s22)
    };

    let (ops3, s22_inv) = invert(s22.as_view(), multiply);
    let (ops4, b12) = {
        // from what I understand negation is basically free as it can be folded into multiplication
        let a11_inv_neg = -&a11_inv;
        let (ops1, mid1) = multiply(a11_inv_neg.as_view(), a12);
        let (ops2, b12) = multiply(mid1.as_view(), s22_inv.as_view());

        (ops1 + ops2, b12)
    };

    // b11 can be reduced to a11_inv - b12 * a21 * a11_inv
    let (ops5, b11) = {
        let (ops1, mid1) = multiply(b12.as_view(), a21);
        let (ops2, mid2) = multiply(mid1.as_view(), a11_inv.as_view());
        let b11 = &a11_inv - mid2;
        let ops3 = split_at * split_at;

        (ops1 + ops2 + ops3, b11)
    };

    let (ops6, b21) = {
        let s22_inv_neg = -&s22_inv;
        let (ops1, mid1) = multiply(s22_inv_neg.as_view(), a21);
        let (ops2, b21) = multiply(mid1.as_view(), a11_inv.as_view());

        (ops1 + ops2, b21)
    };

    let mut res = DMatrix::zeros(n, n);

    res.view_mut((0,0), (split_at,split_at)).copy_from(&b11);
    res.view_mut((0,split_at), (split_at,split_at)).copy_from(&b12);
    res.view_mut((split_at,0), (split_at,split_at)).copy_from(&b21);
    res.view_mut((split_at,split_at), (split_at,split_at)).copy_from(&s22_inv);

    let total_ops = ops1 + ops2 + ops3 + ops4 + ops5 + ops6;

    (total_ops, res)
}

pub fn invert_binet(a: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    invert(a.as_view(), binet::calculate)
}

pub fn invert_strassen(a: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    invert(a.as_view(), strassen::calculate)
}

pub fn invert_laderman(a: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    invert(a.as_view(), laderman::calculate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;
    use crate::util::SquareMatrices;
    
    const EPS: f64 = 1e-4;
    
    fn compare(a: &DMatrix<f64>, b: &DMatrix<f64>) -> bool {
        (a - b).abs().max() <= EPS
    }

    #[quickcheck]
    fn invert_binet_definition(SquareMatrices(a, _, _): SquareMatrices) -> bool {
        compare(&binet::binet(&invert_binet(&a).1, &a).1, &DMatrix::identity(a.nrows(), a.ncols()))
            && compare(&binet::binet(&a, &invert_binet(&a).1).1, &DMatrix::identity(a.nrows(), a.ncols()))
    }

    #[quickcheck]
    fn invert_strassen_definition(SquareMatrices(a, _, _): SquareMatrices) -> bool {
        compare(&binet::binet(&invert_strassen(&a).1, &a).1, &DMatrix::identity(a.nrows(), a.ncols()))
            && compare(&binet::binet(&a, &invert_strassen(&a).1).1, &DMatrix::identity(a.nrows(), a.ncols()))
    }

    #[quickcheck]
    fn invert_laderman_definition(SquareMatrices(a, _, _): SquareMatrices) -> bool {
        compare(&binet::binet(&invert_laderman(&a).1, &a).1, &DMatrix::identity(a.nrows(), a.ncols()))
            && compare(&binet::binet(&a, &invert_laderman(&a).1).1, &DMatrix::identity(a.nrows(), a.ncols()))
    }
}
