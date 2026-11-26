use faer::{Mat, MatRef};
use std::ops::Range;

use crate::{svd::{TruncatedSVD, svd}, util::diag};

const SONS_COUNT: usize = 4;

pub fn split(range: Range<usize>) -> (Range<usize>, Range<usize>) {
    let mid = range.start + range.len() / 2;
    (range.start..mid, mid..range.end)
}

#[derive(Clone, Default)]
pub struct Node {
    pub rows: Range<usize>,
    pub cols: Range<usize>,
    pub kind: NodeKind,
}

impl Node {
    pub fn create_tree(
        a: MatRef<f64>,
        rows: Range<usize>,
        cols: Range<usize>,
        r: usize,
        eps: f64,
    ) -> Self {
        let slice = a.get(rows.clone(), cols.clone());
        if r >= rows.len() || r >= cols.len() {
            let svd = svd(slice, r);
            return Node::compress_matrix(a, rows, cols, &svd, r);
        }
        let svd = svd(slice, r + 1);
        if svd.s[r] < eps {
            return Node::compress_matrix(a, rows, cols, &svd, r);
        }
        let (top_rows, bottom_rows) = split(rows.clone());
        let (left_cols, right_cols) = split(cols.clone());

        Node {
            rows,
            cols,
            kind: NodeKind::Branch([
                Box::new(Node::create_tree(a, top_rows.clone(), left_cols.clone(), r, eps)),
                Box::new(Node::create_tree(a, top_rows.clone(), right_cols.clone(), r, eps)),
                Box::new(Node::create_tree(a, bottom_rows.clone(), left_cols.clone(), r, eps)),
                Box::new(Node::create_tree(a, bottom_rows.clone(), right_cols.clone(), r, eps)),
            ]),
        }
    }

    fn compress_matrix(
        a: MatRef<f64>,
        rows: Range<usize>,
        cols: Range<usize>,
        svd: &TruncatedSVD,
        r: usize,
    ) -> Self {
        let slice = a.get(rows.clone(), cols.clone());
        let is_zero = slice
            .row_iter()
            .all(|row| row.iter().all(|&x| x == 0.0));

        let rank = r
            .min(svd.s.len())
            .min(svd.u.ncols())
            .min(svd.vt.nrows());

        Node {
            rows,
            cols,
            kind: if is_zero {
                NodeKind::Zero
            } else {
                NodeKind::LowRank {
                    u: svd.u.subcols(0, rank).to_owned(),
                    vt: diag(&svd.s[..rank]) * svd.vt.subrows(0, rank),
                }
            },
        }
    }

    pub fn compressed_size_bytes(&self) -> usize {
        match &self.kind {
            NodeKind::Zero => 0,
            NodeKind::LowRank { u, vt } => {
                (u.nrows() * u.ncols() + vt.nrows() * vt.ncols()) * std::mem::size_of::<f64>()
            }
            NodeKind::Branch(sons) => sons.iter().map(|s| s.compressed_size_bytes()).sum(),
        }
    }
}

#[derive(Clone, Default)]
pub enum NodeKind {
    #[default]
    Zero,
    LowRank {
        u: Mat<f64>,
        vt: Mat<f64>,
    },
    Branch([Box<Node>; SONS_COUNT]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;
    use quickcheck_macros::quickcheck;

    fn test_node_r1(node: &Node) -> bool {
        match &node.kind {
            NodeKind::Zero => true,
            NodeKind::LowRank { u, vt } => u.ncols() == 1 && vt.nrows() == 1,
            NodeKind::Branch(sons) => sons.iter().all(|son| test_node_r1(son)),
        }
    }

    #[quickcheck]
    fn square_r1(SquareMatrix(a): SquareMatrix) -> bool {
        let n = a.nrows();
        test_node_r1(&Node::create_tree(a.as_ref(), 0..n, 0..n, 1, 1e-3))
    }
}
