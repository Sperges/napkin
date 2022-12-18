#[derive(Debug)]
pub enum Expr {
	_Func(String, Vec<Expr>),
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
    _Power,
    Modulo,
}

#[derive(Debug)]
pub enum UnaryOp {
    Negative,
    Factorial,
}