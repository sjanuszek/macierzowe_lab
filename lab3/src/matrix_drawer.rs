use bmp::Image;
use faer::{Mat, MatRef};

use crate::tree::Node;

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
    
    draw_channel(r_mat.as_ref(), "r.bmp");
    draw_channel(g_mat.as_ref(), "g.bmp");
    draw_channel(b_mat.as_ref(), "b.bmp");
}

fn draw_channel(a: MatRef<f64>, filename: &str) {
    let tree = Node::create_tree(a, 0..a.nrows(), 0..a.ncols(), 4, 1e-3);
    let mut bitmap = Image::new(a.nrows() as u32, a.ncols() as u32);
    for (x, y) in bitmap.coordinates() {
        bitmap.set_pixel(x, y, bmp::Pixel { r: 255, g: 255, b: 255 });
    }

    bitmap.save(filename);
}