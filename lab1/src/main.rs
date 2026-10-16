mod util;
mod strassen;
mod laderman;
mod binet;

use nalgebra::dmatrix;

fn main() {
    let matrix = dmatrix![
        1./3., 2./3., 1., -1.;
        0., 0., 2.3, 4.5;
        8., 2.1, 3.2, 2.3;
        2., 4., 9., -2.;
    ];
    
    println!("{}", matrix);
    println!("{}", binet::binet(&matrix, &matrix).1);
    println!("{}", strassen::strassen(&matrix, &matrix).1);
    println!("{}", laderman::laderman(&matrix, &matrix).1);
}
