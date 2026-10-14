use std::{fs::File, io::Write, time::Instant};

use csv::Writer;
use serde::Serialize;

use crate::{binet, laderman::laderman, strassen::strassen, util::generate_random_matrix};

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
        let (a, b) = (generate_random_matrix(*size), generate_random_matrix(*size));

        // temp
        //let now = Instant::now();
        //let (bin_ops, _) = binet::mul(&a, &b);
        //let bin_time = now.elapsed().as_secs_f64();
        //wtr.serialize(TestResult {
        //    matrix_size: *size,
        //    algorithm: "Binet".to_string(),
        //    number_of_operations: bin_ops,
        //    time: bin_time,
        //}).expect("FAILED TO WRITE TO CSV");

        let now = Instant::now();
        let (str_ops, _) = strassen(&a, &b);
        let str_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "Strassen".to_string(),
            number_of_operations: str_ops,
            time: str_time,
        }).expect("FAILED TO WRITE TO CSV");
        
        let now = Instant::now();
        let (lad_ops, _) = laderman(&a, &b);
        let lad_time = now.elapsed().as_secs_f64();
        wtr.serialize(TestResult {
            matrix_size: *size,
            algorithm: "Laderman".to_string(),
            number_of_operations: lad_ops,
            time: lad_time,
        }).expect("FAILED TO WRITE TO CSV");

        println!("Matrix size: {}", size);
        println!("----------------------------");
        println!("Strassen (dynamic peeling) number of operations: {}", str_ops);
        println!("Strassen (dynamic peeling) run time: {}ms", str_time);
        println!("----------------------------");

        println!("Laderman (dynamic padding) number of operations: {}", lad_ops);
        println!("Laderman (dynamic padding) run time: {}ms", lad_time);
        println!("----------------------------\n");
    }

    let mut file = File::create("test.csv").expect("FAILED TO CREATE FILE");
    file.write_all(&wtr.into_inner().expect("FAILED TO UNPACK WRITER")).expect("FAILED TO WRITE CSV");
}
