use nalgebra::{DMatrix, DMatrixView};

pub fn generate_random_matrix(n: usize) -> DMatrix<f64> {
    let mut random: Vec<f64> = vec![f64::default(); n * n];
    random.iter_mut()
        .for_each(|val| *val = rand::random_range(0.00000001..1.));
    DMatrix::from_vec(n, n, random)
}

pub fn split_view<'a>(view: &'a DMatrixView<f64>) -> (DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>) {
    let (n, _) = view.shape();
    let split_at = n / 2;

    let a11 = view.view((0,0), (split_at,split_at));
    let a12 = view.view((0,split_at), (split_at,split_at));

    let a21 = view.view((split_at,0), (split_at,split_at));
    let a22 = view.view((split_at,split_at), (split_at,split_at));

    (a11, a12, 
     a21, a22)
}

pub fn split_view_3x3<'a>(view: &'a DMatrixView<f64>) -> (
    DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>,
    DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>,
    DMatrixView<'a, f64>, DMatrixView<'a, f64>, DMatrixView<'a, f64>,
) {
    let (n, _) = view.shape();
    let split_at = n / 3;
    
    let a11 = view.view((0,0), (split_at,split_at));
    let a12 = view.view((0,split_at), (split_at,split_at));
    let a13 = view.view((0,2*split_at), (split_at, split_at));

    let a21 = view.view((split_at,0), (split_at,split_at));
    let a22 = view.view((split_at,split_at), (split_at,split_at));
    let a23 = view.view((split_at,2*split_at), (split_at, split_at));

    let a31 = view.view((2*split_at,0), (split_at,split_at));
    let a32 = view.view((2*split_at,split_at), (split_at,split_at));
    let a33 = view.view((2*split_at,2*split_at), (split_at, split_at));

    (a11, a12, a13,
     a21, a22, a23,
     a31, a32, a33)
}
