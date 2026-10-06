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
    let a12 = view.view((0,split_at), (split_at, split_at));
    let a21 = view.view((split_at,0), (split_at,split_at));
    let a22 = view.view((split_at, split_at), (split_at, split_at));

    (a11, a12, a21, a22)
}
