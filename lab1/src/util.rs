use nalgebra::{DMatrix, DMatrixView};
use quickcheck::Arbitrary;

pub fn generate_random_matrix(n: usize) -> DMatrix<f64> {
    let mut random: Vec<f64> = vec![f64::default(); n * n];
    random.iter_mut()
        .for_each(|val| *val = rand::random_range(0.00000001..1.));
    DMatrix::from_vec(n, n, random)
}

pub fn dynamic_peeling<'a>(a: &'a DMatrixView<f64>, b: &'a DMatrixView<f64>, calculate: fn(DMatrixView<f64>, DMatrixView<f64>) -> (usize, DMatrix<f64>)) -> (usize, DMatrix<f64>) {
    let (n, _) = a.shape();
    let m = n - 1;

    let a11 = a.view((0, 0), (m, m));
    let a12 = a.view((0, m), (m, 1));
    let a21 = a.view((m, 0), (1, m));
    let a22 = a[(m, m)];

    let b11 = b.view((0, 0), (m, m));
    let b12 = b.view((0, m), (m, 1));
    let b21 = b.view((m, 0), (1, m));
    let b22 = b[(m, m)];

    let (ops_core, core) = calculate(a11, b11);

    // (+) m^2 additions, (*) (m x 1 by 1 x m) m^2 multiplications = 2m^2
    let c11 = core + &a12 * &b21;
    // (*) (m x m by m x 1) m^2 multiplications + m^2 - m additions, (+) m additions, (*) (m x 1 * scalar) m multiplications =  2m^2 + m
    let c12 = &a11 * &b12 + &a12 * b22;
    // (*) (1 x m by m x m) m^2 multiplications + m^2 - m additions, (+) m additions, (*) (scalar * 1 x m) m multiplications = 2m^2 + m
    let c21 = &a21 * &b11 + a22 * b21;
    // (*) (1 x m by m x 1) m multiplications + m - 1 additions, (+) 1 addition, (*) (scalar * scalar) 1 multiplications = 2m + 1
    let c22 = (&a21 * &b12)[(0,0)] + &a22 * b22;

    let mut res = DMatrix::zeros(n, n);
    res.view_mut((0, 0), (m, m)).copy_from(&c11);
    res.view_mut((0, m), (m, 1)).copy_from(&c12);
    res.view_mut((m, 0), (1, m)).copy_from(&c21);
    res[(m, m)] = c22;

    let peeling_ops = 6 * m * m + 4 * m + 1;

    return (ops_core + peeling_ops, res);
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

#[derive(Clone, Debug)]
pub struct SquareMatrices(pub DMatrix<f64>, pub DMatrix<f64>, pub DMatrix<f64>);

impl Arbitrary for SquareMatrices {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let size = g.size().isqrt();
        SquareMatrices(generate_random_matrix(size), generate_random_matrix(size), generate_random_matrix(size))
    }
    
    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let size = self.0.ncols() / 2;
        Box::new([
            SquareMatrices(generate_random_matrix(size), generate_random_matrix(size), generate_random_matrix(size))
        ].into_iter())
    }
}
