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
    
    let binet = binet::binet(&matrix, &matrix).1;
    let strassen = strassen::strassen(&matrix, &matrix).1;
    let laderman = laderman::laderman(&matrix, &matrix).1;
    eprintln!("Binet: {:.16}", binet);
    eprintln!("Strassen: {:.16}", strassen);
    eprintln!("Laderman: {:.16}", laderman);
    
    let octave_result = dmatrix![
        6.111111111111111e+00,  -1.677777777777778e+00,  -3.933333333333334e+00,  6.966666666666667e+00;
        2.740000000000000e+01,   2.283000000000000e+01,   4.786000000000000e+01, -3.710000000000001e+00;
        3.286666666666667e+01,   2.125333333333333e+01,   4.377000000000000e+01,  4.210000000000001e+00;
        6.866666666666667e+01,   1.223333333333333e+01,   2.200000000000000e+01,  4.070000000000000e+01;
    ];

    eprintln!("Octave vs Binet: {:.16}", (&octave_result - &binet).abs().max());
    eprintln!("Octave vs Strassen: {:.16}", (&octave_result - &strassen).abs().max());
    eprintln!("Octave vs Laderman: {:.16}", (&octave_result - &laderman).abs().max());
}
