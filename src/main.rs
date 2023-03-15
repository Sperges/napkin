use std::{
    collections::{HashMap, HashSet},
    fs, path::PathBuf,
};

use anyhow::{Error, Result};
use pest::{
    iterators::{Pair, Pairs},
    pratt_parser::PrattParser, Parser,
};

#[derive(clap::Parser, Debug)]
struct Cli {
    path: PathBuf,
}

#[derive(thiserror::Error, Debug)]
enum NapkinError {
    // #[error("Circular reference found {idents:#?}")]
    // CircularReference { idents: Vec<String> },
    // #[error("Function Cell: {function_tag:?} references caller Cell: {caller_tag:?}")]
    // FunctionReferencesCaller { function_tag: String, caller_tag: String },
    #[error("Identifier: {ident:?} in Cell: {tag:?} does not reference a valid cell tag")]
    IdentifierNotFound { ident: String, tag: String },
    #[error("Cell: {tag:?} references itself")]
    CellReferencesSelf { tag: String },
}

#[derive(Debug)]
struct Napkin {
    pub groups: Vec<Group>,
}

impl Napkin {
    // fn cells(&self) -> Vec<&Cell> {
    //     return self.groups.iter().flat_map(|group| group.cell()).collect();
    // }

    fn functions(&self) -> Vec<&Cell> {
        return self
            .groups
            .iter()
            .flat_map(|group| group.function())
            .collect();
    }
}

#[derive(Debug)]
pub enum Group {
    Text(String),
    Cell(Box<Cell>),
}

impl Group {
    fn cell(&self) -> Option<&Cell> {
        match self {
            Group::Text(_) => None,
            Group::Cell(cell) => Some(cell),
        }
    }

    fn function(&self) -> Option<&Cell> {
        match self {
            Group::Text(_) => None,
            Group::Cell(cell) => cell.function(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Cell {
    Hidden {
        tag: String,
        expr: Box<Expr>,
    },
    Visible {
        tag: String,
        expr: Box<Expr>,
    },
    Functional {
        tag: String,
        args: Vec<String>,
        expr: Box<Expr>,
    },
}

impl Cell {
    fn function(&self) -> Option<&Self> {
        return match self {
            Cell::Hidden { tag: _, expr: _ } => None,
            Cell::Visible { tag: _, expr: _ } => None,
            Cell::Functional {
                tag: _,
                args: _,
                expr: _,
            } => Some(self),
        };
    }

    fn tag(&self) -> &String {
        return match self {
            Cell::Hidden {
                tag: ident,
                expr: _,
            } => ident,
            Cell::Visible {
                tag: ident,
                expr: _,
            } => ident,
            Cell::Functional {
                tag: ident,
                args: _,
                expr: _,
            } => ident,
        };
    }

    fn idents(&self) -> Vec<&String> {
        return match self {
            Cell::Hidden { tag: _, expr } => expr.idents(),
            Cell::Visible { tag: _, expr } => expr.idents(),
            Cell::Functional {
                tag: _,
                args: _,
                expr,
            } => expr.idents(),
        };
    }

    fn args(&self) -> Option<Vec<&String>> {
        return match self {
            Cell::Hidden { tag: _, expr: _ } => None,
            Cell::Visible { tag: _, expr: _ } => None,
            Cell::Functional {
                tag: _,
                args,
                expr: _,
            } => Some(args.iter().map(|arg| arg).collect()),
        };
    }

    fn expr(&self) -> Option<&Expr> {
        match self {
            Cell::Hidden { tag: _, expr } => Some(expr),
            Cell::Visible { tag: _, expr } => Some(expr),
            Cell::Functional {
                tag: _,
                args: _,
                expr,
            } => Some(expr),
        }
    }
}

#[derive(Debug, Clone)]
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

impl Expr {
    fn idents(&self) -> Vec<&String> {
        return match self {
            Expr::Func(_, exprs) => exprs.iter().flat_map(|expr| expr.idents()).collect(),
            Expr::Ident(ident) => vec![ident],
            Expr::Number(_) => vec![],
            Expr::Unary { op: _, expr } => expr.idents(),
            Expr::Binary { lhs, op: _, rhs } => [lhs.idents(), rhs.idents()].concat(),
        };
    }
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Negative,
    Factorial,
}

trait PairExt {
    fn as_f64(&self) -> f64;
}

impl PairExt for Pair<'_, Rule> {
    fn as_f64(&self) -> f64 {
        return self.as_str().parse::<f64>().unwrap();
    }
}

trait PairsExt {
    fn as_ident(&mut self) -> String;
    fn as_expr(&mut self) -> Pairs<Rule>;
    fn as_args(&mut self) -> Pairs<Rule>;
}

impl PairsExt for Pairs<'_, Rule> {
    fn as_ident(&mut self) -> String {
        return self.next().unwrap().as_str().to_string();
    }

    fn as_expr(&mut self) -> Pairs<Rule> {
        return self.next().unwrap().into_inner();
    }

    fn as_args(&mut self) -> Pairs<Rule> {
        return self.next().unwrap().into_inner();
    }
}

#[derive(pest_derive::Parser)]
#[grammar = "napkin.pest"]
pub struct NapkinParser {}

lazy_static::lazy_static! {
    static ref EXPRESSION_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        return PrattParser::new()
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(modulo, Left))
            .op(Op::infix(power, Left))
            .op(Op::prefix(negative))
    };
}

#[derive(Debug)]
struct Context<'a> {
    cells: HashMap<&'a String, &'a Expr>,
    functions: HashMap<&'a String, (Vec<&'a String>, &'a Expr)>,
}

fn parse_napkin(pairs: Pairs<Rule>) -> Result<Napkin> {
    return Ok(Napkin {
        groups: parse_groups(pairs)?,
    });
}

fn parse_groups(pairs: Pairs<Rule>) -> Result<Vec<Group>> {
    let mut groups = Vec::new();

    for pair in pairs.into_iter() {
        if let Some(group) = parse_group(pair)? {
            groups.push(group)
        } else {
            break;
        }
    }

    return Ok(groups);
}

fn parse_group(pair: Pair<Rule>) -> Result<Option<Group>> {
    return Ok(match pair.as_rule() {
        Rule::hidden_cell => {
            let mut pairs = pair.into_inner();
            Some(Group::Cell(Box::new(Cell::Hidden {
                tag: pairs.as_ident(),
                expr: Box::new(parse_expr(pairs.as_expr())),
            })))
        }
        Rule::visible_cell => {
            let mut pairs = pair.into_inner();
            Some(Group::Cell(Box::new(Cell::Visible {
                tag: pairs.as_ident(),
                expr: Box::new(parse_expr(pairs.as_expr())),
            })))
        }
        Rule::function_cell => {
            let mut pairs = pair.into_inner();
            Some(Group::Cell(Box::new(Cell::Functional {
                tag: pairs.as_ident(),
                args: parse_args(pairs.as_args())?,
                expr: Box::new(parse_expr(pairs.as_expr())),
            })))
        }
        Rule::text => Some(Group::Text(pair.as_str().to_string())),
        Rule::EOI => None,
        _ => unreachable!(),
    });
}

fn parse_args(pairs: Pairs<Rule>) -> Result<Vec<String>> {
    return Ok(pairs.into_iter().flat_map(|pair| parse_arg(pair)).collect());
}

fn parse_arg(pair: Pair<Rule>) -> Option<String> {
    return match pair.as_rule() {
        Rule::ident => Some(pair.as_str().to_string()),
        _ => unreachable!(),
    };
}

fn parse_expr(pairs: Pairs<Rule>) -> Expr {
    return EXPRESSION_PARSER
        .map_primary(map_primary)
        .map_prefix(map_prefix)
        .map_infix(map_infix)
        .map_postfix(map_postfix)
        .parse(pairs);
}

fn map_primary(primary: Pair<Rule>) -> Expr {
    return match primary.as_rule() {
        Rule::func => {
            let mut func = primary.into_inner();
            Expr::Func(
                func.as_ident(),
                func.into_iter()
                    .map(|arg| Box::new(map_primary(arg)))
                    .collect(),
            )
        }
        Rule::number => Expr::Number(primary.as_f64()),
        Rule::ident => Expr::Ident(primary.as_str().to_string()),
        Rule::expr => parse_expr(primary.into_inner()),
        rule => unreachable!("Expr::parse expected atom, found {:?}", rule),
    };
}

fn map_prefix(op: Pair<Rule>, rhs: Expr) -> Expr {
    return Expr::Unary {
        op: match op.as_rule() {
            Rule::negative => UnaryOp::Negative,
            _ => unreachable!(),
        },
        expr: Box::new(rhs),
    };
}

fn map_infix(lhs: Expr, op: Pair<Rule>, rhs: Expr) -> Expr {
    return Expr::Binary {
        lhs: Box::new(lhs),
        op: match op.as_rule() {
            Rule::add => BinaryOp::Add,
            Rule::subtract => BinaryOp::Subtract,
            Rule::multiply => BinaryOp::Multiply,
            Rule::divide => BinaryOp::Divide,
            Rule::modulo => BinaryOp::Modulo,
            Rule::power => BinaryOp::Power,
            rule => unreachable!("Expr::parse expected infix operation, found {:?}", rule),
        },
        rhs: Box::new(rhs)
    };
}

fn map_postfix(lhs: Expr, op: Pair<Rule>) -> Expr {
    Expr::Unary {
        op: match op.as_rule() {
            Rule::factorial => UnaryOp::Factorial,
            _ => unreachable!(),
        },
        expr: Box::new(lhs),
    }
}

fn analyze(napkin: &Napkin) -> Vec<Error> {
    let mut results = Vec::new();

    let cell_names: HashSet<&String> = napkin
        .groups
        .iter()
        .flat_map(|group| group.cell().map(|cell| cell.tag()))
        .collect();

    for group in napkin.groups.iter() {
        match group {
            Group::Text(_) => continue,
            Group::Cell(cell) => {
                let tag = cell.tag();
                for ident in cell.idents() {
                    if ident == tag {
                        results.push(Error::new(NapkinError::CellReferencesSelf {
                            tag: tag.clone(),
                        }));
                    }
                    // TODO: Cyclical references
                    // TODO: Function references caller
                    if cell_names.contains(&ident) {
                        continue;
                    }
                    if let Some(args) = cell.args() {
                        if args.contains(&ident) {
                            continue;
                        }
                    }
                    results.push(Error::new(NapkinError::IdentifierNotFound {
                        ident: ident.clone(),
                        tag: tag.clone(),
                    }));
                }
            }
        }
    }

    return results;
}

fn evaluate(napkin: &Napkin) -> String {
    let mut strings: Vec<String> = Vec::new();

    let cells: HashMap<&String, &Expr> = napkin
        .groups
        .iter()
        .flat_map(|group| {
            group
                .cell()
                .map(|cell| cell.expr().map(|expr| (cell.tag(), expr)))
        })
        .flatten()
        .collect();

    let functions = napkin
        .functions()
        .iter()
        .map(|cell| (cell.tag(), (cell.args().unwrap(), cell.expr().unwrap())))
        .collect();

    let mut context: Context = Context { cells, functions };

    for group in napkin.groups.iter() {
        match group {
            Group::Text(text) => strings.push(text.clone()),
            Group::Cell(cell) => match cell.as_ref() {
                Cell::Hidden { tag: _, expr: _ } => continue,
                Cell::Visible { tag: _, expr } => {
                    let result = eval_expr(&mut context, expr);
                    strings.push(result.to_string());
                }
                Cell::Functional {
                    tag: _,
                    args: _,
                    expr: _,
                } => continue,
            },
        }
    }

    return strings.concat();
}

fn eval_expr<'a>(context: &mut Context<'a>, expr: &'a Expr) -> f64 {
    match expr {
        Expr::Ident(ident) => eval_ident(context, ident),
        Expr::Number(number) => number.clone(),
        Expr::Unary { op, expr } => eval_unary(context, op, expr),
        Expr::Binary { lhs, op, rhs } => eval_binary(context, lhs, op, rhs),
        Expr::Func(ident, args) => eval_func(context, ident, args),
    }
}

fn eval_ident(context: &mut Context, ident: &String) -> f64 {
    return eval_expr(context, &context.cells[ident]);
}

fn eval_func<'a>(context: &mut Context<'a>, ident: &String, arg_exprs: &'a Vec<Box<Expr>>) -> f64 {
    let (arg_tags, expr) = &context.functions[ident];

    context.cells.extend(
        arg_tags
            .iter()
            .zip(arg_exprs.iter())
            .map(|(tag, expr)| (*tag, expr.as_ref()))
            .collect::<HashMap<&String, &Expr>>(),
    );

    return eval_expr(context, expr);
}

fn eval_unary<'a>(context: &mut Context<'a>, op: &UnaryOp, expr: &'a Expr) -> f64 {
    let expr = eval_expr(context, expr);
    return match op {
        UnaryOp::Negative => -expr,
        UnaryOp::Factorial => 1.0 + expr,//((1..(expr as usize) + 1).product::<usize>()) as f64,
    };
}

fn eval_binary<'a>(context: &mut Context<'a>, lhs: &'a Expr, op: &BinaryOp, rhs: &'a Expr) -> f64 {
    let lhs = eval_expr(context, lhs);
    let rhs = eval_expr(context, rhs);
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Subtract => lhs - rhs,
        BinaryOp::Multiply => lhs * rhs,
        BinaryOp::Divide => lhs / rhs,
        BinaryOp::Power => lhs.powf(rhs),
        BinaryOp::Modulo => lhs % rhs,
    }
}

fn run_cli() -> Result<()> {
    let cli = <Cli as clap::Parser>::parse();
    return run_file(&cli.path)
}

fn run_file(path: &PathBuf) -> Result<()> {
    let file = fs::read_to_string(path)?;
    let pairs = NapkinParser::parse(Rule::file, &file)?;
    let napkin = parse_napkin(pairs)?;
    let errors = analyze(&napkin);
    if errors.len() == 0 {
        println!("{}", evaluate(&napkin));
    } else {
        for error in errors {
            eprintln!("{}", error.to_string());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    run_cli()
}
