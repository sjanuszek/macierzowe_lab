use nalgebra::DMatrix;

pub fn generate_random_matrix(n: usize) -> DMatrix<f64> {
    let mut random: Vec<f64> = vec![f64::default(); n * n];
    random.iter_mut()
        .for_each(|val| *val = rand::random_range(0.00000001..1.));
    DMatrix::from_vec(n, n, random)
}

