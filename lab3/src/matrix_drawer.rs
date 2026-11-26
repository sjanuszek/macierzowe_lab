use std::fs::{self, File};
use std::io::Write;

use bmp::{Image, Pixel};
use faer::{Mat, MatRef};

use crate::{svd::full_singular_values, tree::{Node, NodeKind}};

#[derive(Clone, Copy)]
enum Channel { R, G, B }

pub fn draw_original_and_svd(
    filename: &str,
) -> (Mat<f64>, Mat<f64>, Mat<f64>, [f64; 3], [f64; 3], [f64; 3]) {
    let bitmap = bmp::open(filename).expect("Brak zdjęcia");
    let mut r_mat = Mat::zeros(bitmap.get_width() as usize, bitmap.get_height() as usize);
    let mut g_mat = Mat::zeros(bitmap.get_width() as usize, bitmap.get_height() as usize);
    let mut b_mat = Mat::zeros(bitmap.get_width() as usize, bitmap.get_height() as usize);

    for (x, y) in bitmap.coordinates() {
        let Pixel { r, g, b } = bitmap.get_pixel(x, y);
        r_mat[(x as usize, y as usize)] = r as f64;
        g_mat[(x as usize, y as usize)] = g as f64;
        b_mat[(x as usize, y as usize)] = b as f64;
    }

    fs::create_dir_all("out").unwrap();
    save_raw_channel(r_mat.as_ref(), "out/orig_R.bmp", Channel::R);
    save_raw_channel(g_mat.as_ref(), "out/orig_G.bmp", Channel::G);
    save_raw_channel(b_mat.as_ref(), "out/orig_B.bmp", Channel::B);

    let sr = full_singular_values(r_mat.as_ref());
    let sg = full_singular_values(g_mat.as_ref());
    let sb = full_singular_values(b_mat.as_ref());

    let total = sr.len().max(sg.len()).max(sb.len());
    let mut csv = File::create("out/singular_values.csv").unwrap();
    writeln!(csv, "i,R,G,B").unwrap();
    for i in 0..total {
        let vr = sr.get(i).copied().unwrap_or(0.0);
        let vg = sg.get(i).copied().unwrap_or(0.0);
        let vb = sb.get(i).copied().unwrap_or(0.0);
        writeln!(csv, "{},{},{},{}", i + 1, vr, vg, vb).unwrap();
    }

    let s1 = [sr[0], sg[0], sb[0]];
    let s2k = [
        *sr.last().unwrap_or(&0.0),
        *sg.last().unwrap_or(&0.0),
        *sb.last().unwrap_or(&0.0),
    ];

    let s2k_mid = [
        sr.get(sr.len() / 2).copied().unwrap_or(0.0),
        sg.get(sg.len() / 2).copied().unwrap_or(0.0),
        sb.get(sb.len() / 2).copied().unwrap_or(0.0),
    ];

    (r_mat, g_mat, b_mat, s1, s2k, s2k_mid)
}

pub fn draw_compression_case(
    name: &str,
    r_mat: MatRef<f64>,
    g_mat: MatRef<f64>,
    b_mat: MatRef<f64>,
    r: usize,
    delta: [f64; 3],
) {
    let dir = format!("out/{name}");
    fs::create_dir_all(&dir).unwrap();

    let mut rgb_bitmap = Image::new(r_mat.nrows() as u32, r_mat.ncols() as u32);
    let sz_r = draw_channel(
        r_mat,
        &format!("{dir}/comp_R.bmp"),
        &format!("{dir}/out_R.bmp"),
        &mut rgb_bitmap,
        Channel::R,
        r,
        delta[0],
    );
    let sz_g = draw_channel(
        g_mat,
        &format!("{dir}/comp_G.bmp"),
        &format!("{dir}/out_G.bmp"),
        &mut rgb_bitmap,
        Channel::G,
        r,
        delta[1],
    );
    let sz_b = draw_channel(
        b_mat,
        &format!("{dir}/comp_B.bmp"),
        &format!("{dir}/out_B.bmp"),
        &mut rgb_bitmap,
        Channel::B,
        r,
        delta[2],
    );
    rgb_bitmap.save(format!("{dir}/compressed_RGB.bmp")).unwrap();

    let orig_bytes = r_mat.nrows() * r_mat.ncols() * 3;
    let comp_bytes = sz_r + sz_g + sz_b;
    let ratio = if comp_bytes > 0 {
        orig_bytes as f64 / comp_bytes as f64
    } else {
        0.0
    };
    let pct = (comp_bytes as f64 / orig_bytes as f64) * 100.0;

    println!(
        "{name:24} | comp: {comp_bytes} B (R:{sz_r}, G:{sz_g}, B:{sz_b}) | orig: {orig_bytes} B | {pct:.2}% | ratio: {ratio:.2}x"
    );
}

fn save_raw_channel(a: MatRef<f64>, filename: &str, channel: Channel) {
    let mut bitmap = Image::new(a.nrows() as u32, a.ncols() as u32);
    for (x, y) in bitmap.coordinates() {
        let v = a[(x as usize, y as usize)].clamp(0.0, 255.0) as u8;
        bitmap.set_pixel(x, y, match channel {
            Channel::R => Pixel { r: v, g: 0, b: 0 },
            Channel::G => Pixel { r: 0, g: v, b: 0 },
            Channel::B => Pixel { r: 0, g: 0, b: v },
        });
    }
    bitmap.save(filename).unwrap();
}

fn draw_channel(
    a: MatRef<f64>,
    comp_filename: &str,
    out_filename: &str,
    shared_bitmap: &mut Image,
    channel: Channel,
    r: usize,
    eps: f64,
) -> usize {
    let tree = Node::create_tree(a, 0..a.nrows(), 0..a.ncols(), r, eps);

    let mut comp_bitmap = Image::new(a.nrows() as u32, a.ncols() as u32);
    for (x, y) in comp_bitmap.coordinates() {
        comp_bitmap.set_pixel(x, y, Pixel { r: 255, g: 255, b: 255 });
    }
    draw_tree(&tree, &mut comp_bitmap);
    comp_bitmap.save(comp_filename).unwrap();

    let mut ch_bitmap = Image::new(a.nrows() as u32, a.ncols() as u32);
    draw_compressed(&tree, &mut ch_bitmap, channel);
    ch_bitmap.save(out_filename).unwrap();

    draw_compressed(&tree, shared_bitmap, channel);
    tree.compressed_size_bytes()
}

fn draw_tree(node: &Node, bitmap: &mut Image) {
    match &node.kind {
        NodeKind::Zero => (),
        NodeKind::LowRank { u, vt } => {
            for row in node.rows.clone() {
                for col in node.cols.start..(node.cols.start + u.ncols()) {
                    bitmap.set_pixel(row as u32, col as u32, Pixel { r: 0, g: 0, b: 0 });
                }
            }
            for row in node.rows.start..(node.rows.start + vt.nrows()) {
                for col in node.cols.clone() {
                    bitmap.set_pixel(row as u32, col as u32, Pixel { r: 0, g: 0, b: 0 });
                }
            }
            for row in node.rows.clone() {
                bitmap.set_pixel(row as u32, node.cols.start as u32, Pixel { r: 0, g: 0, b: 0 });
                bitmap.set_pixel(row as u32, (node.cols.end - 1) as u32, Pixel { r: 0, g: 0, b: 0 });
            }
            for col in node.cols.clone() {
                bitmap.set_pixel(node.rows.start as u32, col as u32, Pixel { r: 0, g: 0, b: 0 });
                bitmap.set_pixel((node.rows.end - 1) as u32, col as u32, Pixel { r: 0, g: 0, b: 0 });
            }
        }
        NodeKind::Branch(sons) => {
            for son in sons {
                draw_tree(&son, bitmap);
            }
        }
    }
}

fn draw_compressed(node: &Node, bitmap: &mut Image, channel: Channel) {
    match &node.kind {
        NodeKind::Zero => {},
        NodeKind::LowRank { u, vt } => {
            let a = u * vt;
            for row in 0..a.nrows() {
                for col in 0..a.ncols() {
                    let x = (node.rows.start + row) as u32;
                    let y = (node.cols.start + col) as u32;
                    let value = a[(row, col)] as u8;
                    let Pixel { r, g, b } = bitmap.get_pixel(x, y);
                    bitmap.set_pixel(
                        x,
                        y,
                        match channel {
                            Channel::R => Pixel { r: value, g, b },
                            Channel::G => Pixel { r, g: value, b },
                            Channel::B => Pixel { r, g, b: value },
                        }
                    );
                }
            }
        }
        NodeKind::Branch(sons) => {
            for son in sons {
                draw_compressed(&son, bitmap, channel);
            }
        }
    }
}
