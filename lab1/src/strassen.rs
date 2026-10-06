use nalgebra::{DMatrix, DMatrixView};

fn split_view<'a>(view: &'a DMatrixView<f64>) -> (DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>) {
    let (n, _) = view.shape();
    let split_at = n / 2;

    let a11 = view.view((0,0), (split_at,split_at));
    let a12 = view.view((0,split_at), (split_at, split_at));
    let a21 = view.view((split_at,0), (split_at,split_at));
    let a22 = view.view((split_at, split_at), (split_at, split_at));

    (a11, a12, a21, a22)
}

fn calculate(a_view: DMatrixView<f64>, b_view: DMatrixView<f64>) -> (usize, DMatrix<f64>) {
    let (n, _) = a_view.shape();

    if n == 1 {
        return (1, a_view * b_view);
    }

    if n == 2 {
        let p1 = (a_view[(0,0)] + a_view[(1,1)]) * (b_view[(0, 0)] + b_view[(1, 1)]);
        let p2 = (a_view[(1,0)] + a_view[(1,1)]) * b_view[(0,0)];
        let p3 = a_view[(0,0)] * (b_view[(0,1)] - b_view[(1,1)]);
        let p4 = a_view[(1,1)] * (b_view[(1,0)] - b_view[(0,0)]);
        let p5 = (a_view[(0,0)] + a_view[(0,1)]) * b_view[(1,1)];
        let p6 = (a_view[(1,0)] - a_view[(0,0)]) * (b_view[(0,0)] + b_view[(0,1)]);
        let p7 = (a_view[(0,1)] - a_view[(1,1)]) * (b_view[(1,0)] + b_view[(1,1)]);

        let mut res = DMatrix::zeros(2, 2);
        res[(0,0)] = p1 + p4 - p5 + p7;
        res[(0,1)] = p3 + p5;
        res[(1,0)] = p2 + p4;
        res[(1,1)] = p1 - p2 + p3 + p6;

        return (25, res);
    }
    
    let (a11, a12, a21, a22) = split_view(&a_view);
    let (b11, b12, b21, b22) = split_view(&b_view);

    let (ops1, m1) = calculate((&a11 + &a22).as_view(), (&b11 + &b22).as_view());
    let (ops2, m2) = calculate((&a21 + &a22).as_view(), b11.as_view());
    let (ops3, m3) = calculate(a11.as_view(), (&b12 - &b22).as_view());
    let (ops4, m4) = calculate(a22.as_view(), (&b21 - &b11).as_view());
    let (ops5, m5) = calculate((&a11 + &a12).as_view(), b22.as_view());
    let (ops6, m6) = calculate((&a21 - &a11).as_view(), (&b11 + &b12).as_view());
    let (ops7, m7) = calculate((&a12 - &a22).as_view(), (&b21 + &b22).as_view());

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

pub fn strassen(a: DMatrix<f64>, b: DMatrix<f64>) -> (usize, DMatrix<f64>) {
    let (n, _) = a.shape();

    let a_view = a.view((0, 0), (n, n));
    let b_view = b.view((0, 0), (n, n));

    let (ops, result) = calculate(a_view, b_view);
    (ops, result)
}
