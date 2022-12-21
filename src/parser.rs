use crate::prelude::*;

#[derive(Parser)]
#[grammar = "napkin.pest"]
pub struct NapkinParser {}

lazy_static::lazy_static! {
    static ref EXPRESSION_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        PrattParser::new()
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(modulo, Left))
            .op(Op::infix(power, Left))
            .op(Op::prefix(negative))
    };
}

pub fn parse_napkin(pairs: Pairs<Rule>) -> Napkin {
    Napkin(parse_groups(pairs))
}

fn parse_groups(pairs: Pairs<Rule>) -> Vec<Group> {
    let mut groups = Vec::new();
    for pair in pairs.into_iter() {
        groups.push(match pair.as_rule() {
            Rule::visible_cell => { 
                let mut pairs = pair.into_inner();
                let ident = pairs.next().unwrap().as_str().to_string();
                let expr = Box::new(parse_expr(pairs.next().unwrap().into_inner()));
                Group::Cell(Box::new(Cell::Standard { ident, expr }))
            },
            Rule::function_cell => {
                let mut pairs = pair.into_inner();
                let ident = pairs.next().unwrap().as_str().to_string();
                let args = parse_args(pairs.next().unwrap().into_inner());
                let expr = Box::new(parse_expr(pairs.next().unwrap().into_inner()));
                Group::Cell(Box::new(Cell::Functional { ident, args, expr }))
            },
            Rule::text => Group::Text(pair.as_str().to_string()),
            Rule::EOI => continue,
            _ => unreachable!(),
        });
    }
    groups
}

fn parse_args(pairs: Pairs<Rule>) -> Vec<String> {
    let mut args = Vec::new();
    for pair in pairs.into_iter() {
        args.push(match pair.as_rule() {
            Rule::ident => pair.as_str().to_string(),
            _ => unreachable!(),
        });
    }
    args
}

pub fn parse_expr(pairs: Pairs<Rule>) -> Expr {
    EXPRESSION_PARSER
        .map_primary(map_primary)
        .map_prefix(map_prefix)
        .map_infix(map_infix)
        .map_postfix(map_postfix)
        .parse(pairs)
}

fn map_primary(primary: Pair<Rule>) -> Expr {
    match primary.as_rule() {
		Rule::func => {
            let mut func = primary.into_inner();
            let ident = func.next().unwrap().as_str().to_string();
            let args: Vec<Box<Expr>> = func.into_iter().map(|arg| Box::new(map_primary(arg))).collect();
            Expr::Func(ident, args)
        },
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
        Rule::power  => BinaryOp::Power,
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