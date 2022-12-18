mod parser;
mod ast;

extern crate pest;
#[macro_use]
extern crate pest_derive;

mod prelude {
	pub use crate::*;
	pub use thiserror::Error;
	pub use eyre::{Result, eyre};
	pub use pest::{
		iterators::{Pair, Pairs},
		pratt_parser::PrattParser,
		Parser,
	};
	pub use std::collections::HashMap;
	pub use parser::*;
	pub use ast::*;
}

use std::fs;
use prelude::*;

fn main() -> Result<()> {
    let unparsed_file = fs::read_to_string("test.napkin")?;

    let pairs = NapkinParser::parse(Rule::file, &unparsed_file)?;

	// let state = parse_cells(pairs);

	println!("{:#?}", pairs);
	// println!("{:#?}", cells);

    Ok(())
}

#[derive(Debug)]
pub struct Cell {
	_visible: bool,
	_expr: Box<Expr>,
}

#[derive(Debug, Default)]
pub struct State {
	keys: HashMap<String, usize>,
	exprs: Vec<Box<Expr>>,
	visibles: Vec<bool>,
}

pub fn parse_cells(pairs: Pairs<Rule>) -> State {
	let mut state = State::default();

	for pair in pairs {
		let visible = match pair.as_rule() {
			Rule::visible_cell => true,
			Rule::hidden_cell => false,
			Rule::EOI => continue,
			_ => false,
		};
		let mut pairs = pair.into_inner();
		let key = pairs.next().unwrap().as_str().to_string();
		let expr = parse_expr(pairs.next().unwrap().into_inner());

		state.keys.insert(key.clone(), state.exprs.len());
		state.exprs.push( Box::new(expr));
		state.visibles.push(visible);
	}
	state
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

	#[test]
	fn test_entry() {
		let mut letters = HashMap::new();

		for ch in "a short tratise on fungi".chars() {
			letters.entry(ch).and_modify(|counter| *counter += 1).or_insert(1);
		}

		assert_eq!(letters[&'s'], 2);
		assert_eq!(letters[&'t'], 3);
		assert_eq!(letters[&'u'], 1);
		assert_eq!(letters.get(&'y'), None);
	}
}
