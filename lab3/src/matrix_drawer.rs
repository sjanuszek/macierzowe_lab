use bmp::Image;
use faer::{Mat, MatRef};

use crate::tree::{Node, NodeKind};

pub fn draw_everything(filename: &str) {
    let bitmap = bmp::open(filename).expect("Brak zdjęcia");
    let mut r_mat = Mat::zeros(bitmap.get_width() as usize, bitmap.get_height() as usize);
    let mut g_mat = Mat::zeros(bitmap.get_width() as usize, bitmap.get_height() as usize);
    let mut b_mat = Mat::zeros(bitmap.get_width() as usize, bitmap.get_height() as usize);

    for (x, y) in bitmap.coordinates() {
        let bmp::Pixel { r, g, b } = bitmap.get_pixel(x, y);
        r_mat[(x as usize, y as usize)] = r as f64;
        g_mat[(x as usize, y as usize)] = g as f64;
        b_mat[(x as usize, y as usize)] = b as f64;
    }
    
    let mut bitmap = Image::new(r_mat.nrows() as u32, r_mat.ncols() as u32);
    draw_channel(r_mat.as_ref(), "r.bmp", &mut bitmap, Channel::R);
    draw_channel(g_mat.as_ref(), "g.bmp", &mut bitmap, Channel::G);
    draw_channel(b_mat.as_ref(), "b.bmp", &mut bitmap, Channel::B);
    bitmap.save("compressed.bmp");
}

fn draw_channel(a: MatRef<f64>, filename: &str, shared_bitmap: &mut Image, channel: Channel) {
    let tree = Node::create_tree(a, 0..a.nrows(), 0..a.ncols(), 1, 20.);
    let mut bitmap = Image::new(a.nrows() as u32, a.ncols() as u32);
    for (x, y) in bitmap.coordinates() {
        bitmap.set_pixel(x, y, bmp::Pixel { r: 255, g: 255, b: 255 });
    }
    
    draw_tree(&tree, &mut bitmap);

    bitmap.save(filename);

    draw_compressed(&tree, shared_bitmap, channel);
}

fn draw_tree(node: &Node, bitmap: &mut Image) {
    match &node.kind {
        NodeKind::Zero => (),
        NodeKind::LowRank { u, vt } => {
            for row in node.rows.clone() {
                for col in node.cols.start..(node.cols.start + u.ncols()) {
                    bitmap.set_pixel(row as u32, col as u32, bmp::Pixel { r: 0, g: 0, b: 0 });
                }
            }
            for row in node.rows.start..(node.rows.start + vt.nrows()) {
                for col in node.cols.clone() {
                    bitmap.set_pixel(row as u32, col as u32, bmp::Pixel { r: 0, g: 0, b: 0 });
                }
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
        NodeKind::Zero => (),
        NodeKind::LowRank { u, vt } => {
            let a = u * vt;
            for row in 0..a.nrows() {
                for col in 0..a.ncols() {
                    let x = (node.rows.start + row) as u32;
                    let y = (node.cols.start + col) as u32;
                    let bmp::Pixel { r, g, b } = bitmap.get_pixel(x, y);
                    match channel {
                        Channel::R => bitmap.set_pixel(x, y, bmp::Pixel { r: r + a[(row as usize, col as usize)] as u8, g, b }),
                        Channel::G => bitmap.set_pixel(x, y, bmp::Pixel { r, g: g + a[(row as usize, col as usize)] as u8, b }),
                        Channel::B => bitmap.set_pixel(x, y, bmp::Pixel { r, g, b: b + a[(row as usize, col as usize)] as u8 }),
                    }
                }
            }
        }
        NodeKind::Branch(sons) => {
            for son in sons {
                draw_tree(&son, bitmap);
            }
        }
    }
}

enum Channel { R, G, B }