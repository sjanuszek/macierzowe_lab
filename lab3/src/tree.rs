use faer::{Mat, MatRef};
use std::{array, ops::Range};

use crate::svd::{TruncatedSVD, svd};

const SONS_COUNT: usize = 4;

#[derive(Clone, Copy, Default)]
pub struct IndexRange {
    pub start: usize,
    pub end: usize,
}

impl IndexRange {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn split(&self) -> (Self, Self) {
        let mid = self.start + self.len() / 2;

        (
            Self { start: self.start, end: mid },
            Self { start: mid, end: self.end },
        )
    }

    pub fn get_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

#[derive(Clone, Default)]
pub struct Node {
    rows: IndexRange,
    cols: IndexRange,
    kind: NodeKind,
}

impl Node {
    pub fn create_tree(a: MatRef<f64>, rows: IndexRange, cols: IndexRange, r: usize, eps: f64) -> Self {
        let svd = svd(a, r + 1);
        let v: Node;
        if svd.s[r] < eps {
            v = Node::compress_matrix(a, rows, cols, &svd, r);   
        } else {
            let (top_rows, bottom_rows) = rows.split();
            let (left_cols, right_cols) = cols.split();

            let top_left = a.get(top_rows.get_range(), left_cols.get_range());
            let top_right = a.get(top_rows.get_range(), right_cols.get_range());
            let bottom_left = a.get(bottom_rows.get_range(), left_cols.get_range());
            let bottom_right = a.get(bottom_rows.get_range(), right_cols.get_range());

            let mut sons: [Box<Node>; SONS_COUNT] = array::from_fn(|_| Box::new(Node::default()));
            sons[0] = Box::new(Node::create_tree(top_left, top_rows, left_cols, r, eps));
            sons[1] = Box::new(Node::create_tree(top_right, top_rows, right_cols, r, eps));
            sons[2] = Box::new(Node::create_tree(bottom_left, bottom_rows, left_cols, r, eps));
            sons[3] = Box::new(Node::create_tree(bottom_right, bottom_rows, right_cols, r, eps));

            v = Node::new(rows, cols, NodeKind::Branch(sons))
        }

        return v
    }

    fn new(rows: IndexRange, cols: IndexRange, kind: NodeKind) -> Self {
        Self {
            rows,
            cols,
            kind
        }
    }

    fn all_zero(a: MatRef<f64>) -> bool {
        let mut zero = true;

        a.for_each(|x| {
            if *x != 0. {
                zero = false;
            }
        });

        zero
    }

    fn compress_matrix(a: MatRef<f64>, rows: IndexRange, cols: IndexRange, svd: &TruncatedSVD, r: usize) -> Self {
        if Node::all_zero(a) {
            return Node::new(rows, cols, NodeKind::Zero);
        }
        return Node::new(rows, cols, NodeKind::LowRank { u: svd.u.subcols(0, r).to_owned(), s: svd.s[..r].to_owned(), vt: svd.vt.subrows(0, r).to_owned() });
    }
}

#[derive(Clone, Default)]
pub enum NodeKind {
    #[default]
    Zero,
    LowRank {
        u: Mat<f64>,
        s: Vec<f64>,
        vt: Mat<f64>,
    },
    Branch ([Box<Node>; SONS_COUNT])
}
