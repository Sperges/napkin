use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct Napkin(pub Vec<Group>);

impl Deref for Napkin {
    type Target = Vec<Group>;
    fn deref(&self) -> &Vec<Group> {
        &self.0
    }
}

impl DerefMut for Napkin {
    fn deref_mut(&mut self) -> &mut Vec<Group> {
        &mut self.0
    }
}

#[derive(Debug)]
pub enum Group {
    Text(String),
    Cell(Box<Cell>),
}

#[derive(Debug)]
pub enum Cell {
    Standard{
        ident: String, 
        expr: Box<Expr>,
    },
    Functional {
        ident: String,
        args: Vec<String>,
        expr: Box<Expr>,
    },
}

#[derive(Debug)]
pub enum Expr {
	Func(String, Vec<Box<Expr>>),
    Ident(String),
    Number(f64),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        lhs: Box<Expr>,
        op: BinaryOp,
        rhs: Box<Expr>,
    },
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,
}

#[derive(Debug)]
pub enum UnaryOp {
    Negative,
    Factorial,
}