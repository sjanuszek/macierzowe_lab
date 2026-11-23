use faer::{Mat, MatRef, rand, stats::prelude::StandardNormal};
use rand::prelude::*;

const POWER_ITER: usize = 2;
const OVERSAMPLING: usize = 5;

pub struct TruncatedSVD {
    pub u: Mat<f64>,
    pub s: Vec<f64>,
    pub vt: Mat<f64>
}

// implementation of a probabilistic svd decomposition algorithm as according to:
// Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions by Halko, Martinsson, Tropp 
pub fn svd(matrix: MatRef<f64>, k: usize) -> TruncatedSVD {
    let l = k + OVERSAMPLING;
    let mut rng = rand::rng();
    // draw an n × l Gaussian random matrix omega
    let omega: Mat<f64> = Mat::from_fn(matrix.ncols(), l, |_, _| rng.sample(StandardNormal));

    // form the m x l matrix y = matrix * omega
    let mut y = matrix * &omega;

    // power iterations improve accuracy of the final result
    // multiply y first by matrix then by matrix_transposed
    // calculate q using qr decomposition to protect against round-off errors
    // 2 power iteration series are enough to construct an accurate y matrix
    for _ in 0..POWER_ITER {
        let q = y.qr().compute_thin_Q();

        let z = matrix.transpose() * &q;
        let z = z.qr().compute_thin_Q();

        y = matrix * &z;
    }

    // calculate q of y using qr decomposition
    let q = y.qr().compute_thin_Q();

    // form the matrix b = q_transposed * matrix
    let b = q.transpose() * matrix;

    // now we calculate svd of the dense martix b which will be used to approximate svd of matrix
    let b_svd = b.thin_svd().unwrap();

    // approximate components of svd of matrix
    let u_wave = b_svd.U().subcols(0, k);
    let sigma = b_svd.S();
    let vt = b_svd.V().subcols(0, k).transpose().to_owned();

    let u = q * u_wave;

    let s: Vec<f64> = (0..k).map(|i| sigma[i]).collect();

    TruncatedSVD {u, s, vt}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{testing::*, util::diag};
    use quickcheck_macros::quickcheck;
    
    #[quickcheck]
    fn square_full_svd(SquareMatrix(a): SquareMatrix) -> bool {
        let n = a.nrows();
        let TruncatedSVD { u, s, vt } = svd(a.as_ref(), n);
        u.shape() == (n, n) && s.len() == n && vt.shape() == (n, n)
            && is_close((&u * diag(&s) * &vt).as_ref(), a.as_ref())
            && is_close((&u * u.transpose()).as_ref(), Mat::identity(n, n).as_ref())
            && is_close((&vt * vt.transpose()).as_ref(), Mat::identity(n, n).as_ref())
            && s.iter().all(|&sigma| sigma >= 0.)
    }
    
    #[quickcheck]
    fn square_truncated_svd(SquareMatrix(a): SquareMatrix) -> bool {
        let n = a.nrows();
        let k = n / 2;
        let TruncatedSVD { u, s, vt } = svd(a.as_ref(), k);
        u.shape() == (n, k) && s.len() == k && vt.shape() == (k, n)
            && s.iter().all(|&sigma| sigma >= 0.)
    }
}
