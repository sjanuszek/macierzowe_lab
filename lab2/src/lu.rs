use nalgebra::{DMatrix, DMatrixView, dmatrix};

use crate::{{binet, invert::invert, strassen}, util::split_view};

fn lu_peeling<'a>(
    a: &'a DMatrixView<f64>, 
    multiply: fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>),
    invert: fn(DMatrixView<f64>, fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>)) -> (usize, DMatrix<f64>)
) -> (usize, DMatrix<f64>, DMatrix<f64>) {
    let (n, _) = a.shape();
    let m = n - 1;

    let a11 = a.view((0, 0), (m, m));
    let a12 = a.view((0, m), (m, 1));
    let a21 = a.view((m, 0), (1, m));
    let a22 = a[(m, m)];

    let (ops_core, l11, u11) = lu(a11, multiply, invert);
    let (ops1, u11_inv) = invert(u11.as_view(), multiply);
    let (ops2, l11_inv) = invert(l11.as_view(), multiply);

    let l21 = a21 * &u11_inv;
    let u12 = &l11_inv * a12;
    let s = a22 - (&l21 * &u12)[(0,0)];

    let mut l = DMatrix::zeros(n, n);
    let mut u = DMatrix::zeros(n, n);

    l.view_mut((0, 0), (m, m)).copy_from(&l11);
    l.view_mut((m, 0), (1, m)).copy_from(&l21);
    l[(m, m)] = 1.;

    u.view_mut((0, 0), (m, m)).copy_from(&u11);
    u.view_mut((0, m), (m, 1)).copy_from(&u12);
    u[(m, m)] = s;

    let peeling_ops = 4 * m * m;

    (ops_core + ops1 + ops2 + peeling_ops, l, u)
}

fn lu(
    a: DMatrixView<f64>, 
    multiply: fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>), 
    invert: fn(DMatrixView<f64>, fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>)) -> (usize, DMatrix<f64>), 
) -> (usize, DMatrix<f64>, DMatrix<f64>) {
    let (n, _) = a.shape();

    if n == 1 {
        return (0, dmatrix![1.0], dmatrix![a[(0, 0)]]);
    }

    if n % 2 != 0 { return lu_peeling(&a, multiply, invert); }

    let split_at = n / 2;
    let (a11, a12, 
         a21, a22) = split_view(&a);

    let (ops1, l11, u11) = lu(a11, multiply, invert);
    let (ops2, u11_inv) = invert(u11.as_view(), multiply);
    let (ops3, l21) = multiply(a21, u11_inv.as_view());
    let (ops4, l11_inv) = invert(l11.as_view(), multiply);
    let (ops5, u12) = multiply(l11_inv.as_view(), a12);

    // s can be reduced to a22 - l21 * l11_inv * a12
    let (ops6, s) = {
        let (ops1, mid1) = multiply(l21.as_view(), l11_inv.as_view());
        let (ops2, mid2) = multiply(mid1.as_view(), a12);
        let s = a22 - mid2;
        let ops3 = split_at * split_at;

        (ops1 + ops2 + ops3, s)
    };

    let (ops7, ls, us) = lu(s.as_view(), multiply, invert);

    let mut l = DMatrix::zeros(n, n);
    let mut u = DMatrix::zeros(n, n);

    l.view_mut((0,0), (split_at,split_at)).copy_from(&l11);
    l.view_mut((split_at,0), (split_at,split_at)).copy_from(&l21);
    l.view_mut((split_at,split_at), (split_at,split_at)).copy_from(&ls);

    u.view_mut((0,0), (split_at,split_at)).copy_from(&u11);
    u.view_mut((0,split_at), (split_at,split_at)).copy_from(&u12);
    u.view_mut((split_at,split_at), (split_at,split_at)).copy_from(&us);

    let total_ops = ops1 + ops2 + ops3 + ops4 + ops5 + ops6 + ops7;

    (total_ops, l, u)
}

pub fn lu_binet(a: &DMatrix<f64>) -> (usize, DMatrix<f64>, DMatrix<f64>) {
    lu(a.as_view(), binet::calculate, invert)
}

pub fn lu_strassen(a: &DMatrix<f64>) -> (usize, DMatrix<f64>, DMatrix<f64>) {
    lu(a.as_view(), strassen::calculate, invert)
}
