use faer::diag::Diag;

pub fn diag(v: &Vec<f64>) -> Diag<f64> {
    let mut diag = Diag::zeros(v.len());
    for i in 0..v.len() {
        diag[i] = v[i];
    }
    diag
}