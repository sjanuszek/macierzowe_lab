use nalgebra::{DMatrix, DMatrixView, dmatrix};

use crate::util::{dynamic_peeling, split_view_3x3};

fn calculate(a: DMatrixView<f64>, b: DMatrixView<f64>) -> (usize, DMatrix<f64>) {
    let (n, _) = a.shape();

    if n == 1 {
        return (1, dmatrix![a[(0, 0)] * b[(0, 0)]]);
    }

    if n % 3 != 0 { return dynamic_peeling(&a, &b, calculate); }
    
    let (a11, a12, a13,
         a21, a22, a23,
         a31, a32, a33) = split_view_3x3(&a);

    let (b11, b12, b13,
         b21, b22, b23,
         b31, b32, b33) = split_view_3x3(&b);

    let (ops1, m1) = calculate((a11 + a12 + a13 - a21 - a22 - a32 - a33).as_view(),  b22.as_view());
    let (ops2, m2) = calculate((a11 - a21).as_view(), (-b12 + b22).as_view());
    let (ops3, m3) = calculate(a22.as_view(), (-b11 + b12 + b21 - b22 - b23 - b31 + b33).as_view());
    let (ops4, m4) = calculate((-a11 + a21 + a22).as_view(), (b11 - b12 + b22).as_view());
    let (ops5, m5) = calculate((a21 + a22).as_view(), (-b11 + b12).as_view());
    let (ops6, m6) = calculate(a11, b11);
    let (ops7, m7) = calculate((-a11 + a31 + a32).as_view(), (b11 - b13 + b23).as_view());
    let (ops8, m8) = calculate((-a11 + a31).as_view(), (b13 - b23).as_view());
    let (ops9, m9) = calculate((a31 + a32).as_view(), (-b11 + b13).as_view());
    let (ops10, m10) = calculate((a11 + a12 + a13 - a22 - a23 - a31 - a32).as_view(), b23.as_view());
    let (ops11, m11) = calculate(a32.as_view(), (-b11 + b13 + b21 - b22 - b23 - b31 + b32).as_view());
    let (ops12, m12) = calculate((-a13 + a32 + a33).as_view(), (b22 + b31 - b32).as_view());
    let (ops13, m13) = calculate((a13 - a33).as_view(), (b22 - b32).as_view());
    let (ops14, m14) = calculate(a13, b31);
    let (ops15, m15) = calculate((a32 + a33).as_view(), (-b31 + b32).as_view());
    let (ops16, m16) = calculate((-a13 + a22 + a23).as_view(), (b23 + b31 - b33).as_view());
    let (ops17, m17) = calculate((a13 - a23).as_view(), (b23 - b33).as_view());
    let (ops18, m18) = calculate((a22 + a23).as_view(), (-b31 + b33).as_view());
    let (ops19, m19) = calculate(a12, b21);
    let (ops20, m20) = calculate(a23, b32);
    let (ops21, m21) = calculate(a21, b13);
    let (ops22, m22) = calculate(a31, b12);
    let (ops23, m23) = calculate(a33, b33);

    let c11 = &m6 + &m14 + &m19;
    let c12 = &m1 + &m4 + &m5 + &m6 + &m12 + &m14 + &m15;
    let c13 = &m6 + &m7 + &m9 + &m10 + &m14 + &m16 + &m18;

    let c21 = &m2 + &m3 + &m4 + &m6 + &m14 + &m16 + &m17;
    let c22 = &m2 + &m4 + &m5 + &m6 + &m20;
    let c23 = &m14 + &m16 + &m17 + &m18 + &m21;

    let c31 = &m6 + &m7 + &m8 + &m11 + &m12 + &m13 + &m14;
    let c32 = &m12 + &m13 + &m14 + &m15 + &m22;
    let c33 = &m6 + &m7 + &m8 + &m9 + &m23;

    let split_at = n / 3;
    // the total number of block operations for generating m and c matrixes is 98 (additions and
    // subtractions) and each of these matrixes has (n/3)^2 elements
    let matrix_ops = 98 * split_at * split_at;
    let mut res = DMatrix::zeros(n, n);

    res.view_mut((0,0), (split_at,split_at)).copy_from(&c11);
    res.view_mut((0,split_at), (split_at,split_at)).copy_from(&c12);
    res.view_mut((0,2*split_at), (split_at,split_at)).copy_from(&c13);

    res.view_mut((split_at,0), (split_at,split_at)).copy_from(&c21);
    res.view_mut((split_at,split_at), (split_at,split_at)).copy_from(&c22);
    res.view_mut((split_at,2*split_at), (split_at,split_at)).copy_from(&c23);

    res.view_mut((2*split_at,0), (split_at,split_at)).copy_from(&c31);
    res.view_mut((2*split_at,split_at), (split_at,split_at)).copy_from(&c32);
    res.view_mut((2*split_at,2*split_at), (split_at,split_at)).copy_from(&c33);

    let total_ops = ops1 + ops2 + ops3 + ops4 + ops5 + ops6 + ops7 + ops8 + ops9 + ops10 + ops11 + ops12 + ops13 + ops14 + ops15 + ops16 + ops17 + ops18 + ops19 + ops20 + ops21 + ops22 + ops23 + matrix_ops;

    (total_ops, res)
}

pub fn laderman(a: &DMatrix<f64>, b: &DMatrix<f64>) -> (usize, DMatrix<f64>) {
    calculate(a.as_view(), b.as_view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::SquareMatrices;
    use quickcheck_macros::quickcheck;
    
    #[test]
    fn mul1x1() {
        let (_, res) = laderman(&dmatrix![2.], &dmatrix![3.]);
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
        let (_, res) = laderman(&a, &b);

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
            &laderman(&laderman(&a, &b).1, &c).1,
            &laderman(&a, &laderman(&b, &c).1).1
        )
    }
    
    #[quickcheck]
    fn left_distributivity(SquareMatrices(a, b, c): SquareMatrices) -> bool {
        compare(
            &laderman(&a, &(&b + &c)).1,
            &(laderman(&a, &b).1 + laderman(&a, &c).1)
        )
    }
    
    #[quickcheck]
    fn right_distributivity(SquareMatrices(a, b, c): SquareMatrices) -> bool {
        compare(
            &laderman(&(&b + &c), &a).1,
            &(laderman(&b, &a).1 + laderman(&c, &a).1)
        )
    }
    
    #[quickcheck]
    fn left_identity(SquareMatrices(a, mut i, _): SquareMatrices) -> bool {
        i.fill_with_identity();
        compare(&laderman(&i, &a).1, &a)
    }
    
    #[quickcheck]
    fn right_identity(SquareMatrices(a, mut i, _): SquareMatrices) -> bool {
        i.fill_with_identity();
        compare(&laderman(&a, &i).1, &a)
    }
    
    #[quickcheck]
    fn left_zero(SquareMatrices(a, mut zero, _): SquareMatrices) -> bool {
        zero.fill(0.);
        compare(&laderman(&zero, &a).1, &zero)
    }
    
    #[quickcheck]
    fn right_zero(SquareMatrices(a, mut zero, _): SquareMatrices) -> bool {
        zero.fill(0.);
        compare(&laderman(&a, &zero).1, &zero)
    }
}

