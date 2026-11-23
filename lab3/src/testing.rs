use std::iter;

use faer::{Mat, MatRef};
use quickcheck::Arbitrary;

#[derive(Clone, Debug)]
pub struct SquareMatrix(pub Mat<f64>);

pub const EPS: f64 = 1e-9;

pub fn is_close(a: MatRef<f64>, b: MatRef<f64>) -> bool {
    (a - b).norm_max() <= EPS
}

impl Arbitrary for SquareMatrix {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let n = g.size().isqrt();
        SquareMatrix(Mat::from_fn(n, n, |_, _| rand::random_range(0.001..1000.)))
    }
    
    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let old_size = self.0.ncols();
        if old_size == 1 {
            Box::new(iter::empty())
        } else {
            let n = (old_size / 2).isqrt();
            Box::new([
                SquareMatrix(Mat::from_fn(n, n, |_, _| rand::random_range(0.001..1000.)))
            ].into_iter())
        }
    }
}