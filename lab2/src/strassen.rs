use nalgebra::{DMatrix, DMatrixView, dmatrix};

use crate::util::{dynamic_peeling, split_view};

pub(super) fn calculate(a: DMatrixView<f64>, b: DMatrixView<f64>) -> (usize, DMatrix<f64>) {
    let (n, _) = a.shape();

    if n == 1 {
        return (1, dmatrix![a[(0, 0)] * b[(0, 0)]]);
    }

    // dynamic peeling
    if n % 2 != 0 { return dynamic_peeling(&a, &b, calculate); }
    
    let (a11, a12, 
         a21, a22) = split_view(&a);
    let (b11, b12, 
         b21, b22) = split_view(&b);

    let (ops1, m1) = calculate((a11 + a22).as_view(), (b11 + b22).as_view());
    let (ops2, m2) = calculate((a21 + a22).as_view(), b11.as_view());
    let (ops3, m3) = calculate(a11.as_view(), (b12 - b22).as_view());
    let (ops4, m4) = calculate(a22.as_view(), (b21 - b11).as_view());
    let (ops5, m5) = calculate((a11 + a12).as_view(), b22.as_view());
    let (ops6, m6) = calculate((a21 - a11).as_view(), (b11 + b12).as_view());
    let (ops7, m7) = calculate((a12 - a22).as_view(), (b21 + b22).as_view());

    let c11 = &m1 + &m4 - &m5 + &m7;
    let c12 = &m3 + &m5;
    let c21 = &m2 + &m4;
    let c22 = &m1 - &m2 + &m3 + &m6;

    let split_at = n / 2;
    // the total number of block operations for generating m and c matrixes is 18 (additions and
    // subtractions) and each of these matrixes has (n/2)^2 elements
    let matrix_ops = 18 * split_at * split_at;
    let mut res = DMatrix::zeros(n, n);

    res.view_mut((0,0), (split_at,split_at)).copy_from(&c11);
    res.view_mut((0,split_at), (split_at,split_at)).copy_from(&c12);
    res.view_mut((split_at,0), (split_at,split_at)).copy_from(&c21);
    res.view_mut((split_at,split_at), (split_at,split_at)).copy_from(&c22);

    let total_ops = ops1 + ops2 + ops3 + ops4 + ops5 + ops6 + ops7 + matrix_ops;

    (total_ops, res)
}

pub fn strassen(a: &DMatrix<f64>, b: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    calculate(a.as_view(), b.as_view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::SquareMatrices;
    use quickcheck_macros::quickcheck;
    
    #[test]
    fn mul1x1() {
        let (_, res) = strassen(&dmatrix![2.], &dmatrix![3.]);
        assert_eq!(dmatrix![6.], res);
    }
    
    #[test]
    fn mul2x2() {
        let a = dmatrix![
            1., 2.;
            3., 4.;
        ];
        let b = dmatrix![
            5., 6.;
            7., 8.;
        ];
        let (_, res) = strassen(&a, &b);

        assert_eq!(
            dmatrix![
                19., 22.;
                43., 50.;
            ],
            res
        );
    }
    
    const EPS: f64 = 1e-9;
    
    fn compare(a: &DMatrix<f64>, b: &DMatrix<f64>) -> bool {
        (a - b).abs().max() <= EPS
    }
    
    #[quickcheck]
    fn associativity(SquareMatrices(a, b, c): SquareMatrices) -> bool {
        compare(
            &strassen(&strassen(&a, &b).1, &c).1,
            &strassen(&a, &strassen(&b, &c).1).1
        )
    }
    
    #[quickcheck]
    fn left_distributivity(SquareMatrices(a, b, c): SquareMatrices) -> bool {
        compare(
            &strassen(&a, &(&b + &c)).1,
            &(strassen(&a, &b).1 + strassen(&a, &c).1)
        )
    }
    
    #[quickcheck]
    fn right_distributivity(SquareMatrices(a, b, c): SquareMatrices) -> bool {
        compare(
            &strassen(&(&b + &c), &a).1,
            &(strassen(&b, &a).1 + strassen(&c, &a).1)
        )
    }
    
    #[quickcheck]
    fn left_identity(SquareMatrices(a, mut i, _): SquareMatrices) -> bool {
        i.fill_with_identity();
        compare(&strassen(&i, &a).1, &a)
    }
    
    #[quickcheck]
    fn right_identity(SquareMatrices(a, mut i, _): SquareMatrices) -> bool {
        i.fill_with_identity();
        compare(&strassen(&a, &i).1, &a)
    }
    
    #[quickcheck]
    fn left_zero(SquareMatrices(a, mut zero, _): SquareMatrices) -> bool {
        zero.fill(0.);
        compare(&strassen(&zero, &a).1, &zero)
    }
    
    #[quickcheck]
    fn right_zero(SquareMatrices(a, mut zero, _): SquareMatrices) -> bool {
        zero.fill(0.);
        compare(&strassen(&a, &zero).1, &zero)
    }
}
