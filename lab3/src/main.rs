use crate::matrix_drawer::{draw_compression_case, draw_original_and_svd};

mod tree;
mod svd;
mod testing;
mod util;
mod matrix_drawer;

fn main() {
    let filename = std::env::args().nth(1).unwrap();

    let (r_mat, g_mat, b_mat, s1, s2k, s2k_mid) = draw_original_and_svd(&filename);

    let cases: [(&str, usize, [f64; 3]); 6] = [
        ("r1_delta_s1", 1, s1),
        ("r1_delta_s2k", 1, s2k),
        ("r1_delta_s2k_mid", 1, s2k_mid),
        ("r4_delta_s1", 4, s1),
        ("r4_delta_s2k", 4, s2k),
        ("r4_delta_s2k_mid", 4, s2k_mid),
    ];

    for (name, r, delta) in cases {
        draw_compression_case(name, r_mat.as_ref(), g_mat.as_ref(), b_mat.as_ref(), r, delta);
    }
}
