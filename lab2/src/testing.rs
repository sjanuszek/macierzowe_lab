use std::{fs::File, io::Write, time::Instant};

use csv::Writer;
use serde::Serialize;

use crate::{binet, invert::{invert_binet, invert_laderman, invert_strassen}, laderman::laderman, lu::{lu_binet, lu_strassen}, strassen::strassen, util::generate_random_matrix};

#[derive(Serialize)]
struct TestResult {
    matrix_size: usize,
    algorithm: String,
    number_of_operations: usize,
    time: f64
}

pub fn run_tests(sizes: &[usize]) {
    let mut wtr = Writer::from_writer(vec![]);
    for size in sizes {
        let a = generate_random_matrix(*size);

        let now = Instant::now();
        let (inv_bin_ops, _) = invert_binet(&a);
        let inv_bin_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "Invert Binet".to_string(),
            number_of_operations: inv_bin_ops,
            time: inv_bin_time,
        }).expect("FAILED TO WRITE TO CSV");

        let now = Instant::now();
        let (inv_str_ops, _) = invert_strassen(&a);
        let inv_str_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "Invert Strassen".to_string(),
            number_of_operations: inv_str_ops,
            time: inv_str_time,
        }).expect("FAILED TO WRITE TO CSV");

        let now = Instant::now();
        let (inv_lad_ops, _) = invert_laderman(&a);
        let inv_lad_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "Invert Laderman".to_string(),
            number_of_operations: inv_lad_ops,
            time: inv_lad_time,
        }).expect("FAILED TO WRITE TO CSV");

        let now = Instant::now();
        let (lu_bin_ops, _, _) = lu_binet(&a);
        let lu_bin_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "LU Binet".to_string(),
            number_of_operations: lu_bin_ops,
            time: lu_bin_time,
        }).expect("FAILED TO WRITE TO CSV");

        let now = Instant::now();
        let (lu_str_ops, _, _) = lu_strassen(&a);
        let lu_str_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "LU Strassen".to_string(),
            number_of_operations: lu_str_ops,
            time: lu_str_time,
        }).expect("FAILED TO WRITE TO CSV");

        println!("Matrix size: {}", size);
        println!("----------------------------");

        println!("Invert Binet number of operations: {}", inv_bin_ops);
        println!("Invert Binet run time: {}ms", inv_bin_time);
        println!("----------------------------");

        println!("Invert Strassen number of operations: {}", inv_str_ops);
        println!("Invert Strassen run time: {}ms", inv_str_time);
        println!("----------------------------");

        println!("Invert Laderman number of operations: {}", inv_lad_ops);
        println!("Invert Laderman run time: {}ms", inv_lad_time);
        println!("----------------------------");

        println!("LU Binet number of operations: {}", lu_bin_ops);
        println!("LU Binet run time: {}ms", lu_bin_time);
        println!("----------------------------");

        println!("LU Strassen number of operations: {}", lu_str_ops);
        println!("LU Strassen run time: {}ms", lu_str_time);
        println!("----------------------------\n");
    }

    let mut file = File::create("test.csv").expect("FAILED TO CREATE FILE");
    file.write_all(&wtr.into_inner().expect("FAILED TO UNPACK WRITER")).expect("FAILED TO WRITE CSV");
}
