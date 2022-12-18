use crate::prelude::*;

#[derive(Parser)]
#[grammar = "napkin.pest"]
pub struct NapkinParser {}

lazy_static::lazy_static! {
    static ref NAPKIN_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        PrattParser::new()
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(modulo, Left))
            .op(Op::infix(power, Left))
            .op(Op::prefix(negative))
    };
}

pub fn parse_expr(pairs: Pairs<Rule>) -> Expr {
    NAPKIN_PARSER
        .map_primary(map_primary)
        .map_prefix(map_prefix)
        .map_infix(map_infix)
        .map_postfix(map_postfix)
        .parse(pairs)
}

fn map_primary(primary: Pair<Rule>) -> Expr {
    match primary.as_rule() {
		Rule::func => todo!(),
        Rule::number => Expr::Number(primary.as_str().parse::<f64>().unwrap()),
		Rule::ident => Expr::Ident(primary.as_str().to_string()),
        Rule::expr => parse_expr(primary.into_inner()),
        rule => unreachable!("Expr::parse expected atom, found {:?}", rule),
    }
}

fn map_prefix(op: Pair<Rule>, rhs: Expr) -> Expr {
    let op = match op.as_rule() {
        Rule::negative => UnaryOp::Negative,
        _ => unreachable!(),
    };
    Expr::Unary {
        op,
        expr: Box::new(rhs),
    }
}

fn map_infix(lhs: Expr, op: Pair<Rule>, rhs: Expr) -> Expr {
    let op = match op.as_rule() {
        Rule::add => BinaryOp::Add,
        Rule::subtract => BinaryOp::Subtract,
        Rule::multiply => BinaryOp::Multiply,
        Rule::divide => BinaryOp::Divide,
        Rule::modulo => BinaryOp::Modulo,
        // Rule::pow  => (1..rhs+1).map(|_| lhs).product(),
        rule => unreachable!("Expr::parse expected infix operation, found {:?}", rule),
    };
    Expr::Binary {
        lhs: Box::new(lhs),
        op,
        rhs: Box::new(rhs),
    }
}

fn map_postfix(lhs: Expr, op: Pair<Rule>) -> Expr {
    let op = match op.as_rule() {
        Rule::factorial => UnaryOp::Factorial,
        _ => unreachable!(),
    };
    Expr::Unary {
        op,
        expr: Box::new(lhs),
    }
}