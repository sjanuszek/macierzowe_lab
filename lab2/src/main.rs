use nalgebra::dmatrix;

mod util;
mod binet;
mod strassen;
mod laderman;
mod invert;
mod lu;

fn main() {
    let a = dmatrix![1., 2.; 4., 5.];
    println!("invert_binet:{}", invert::invert_binet(&a).1);
    println!("invert_strassen:{}", invert::invert_strassen(&a).1);
    println!("invert_laderman:{}", invert::invert_laderman(&a).1);

    let (_, l, u) = lu::lu_binet(&a);
    println!("lu_binet:\n L = {}U = {}", l, u);
    let (_, l, u) = lu::lu_strassen(&a);
    println!("lu_strassen:\n L = {}U = {}", l, u);
}
