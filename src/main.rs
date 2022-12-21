mod parser;
mod ast;
mod analyzer;

extern crate pest;
#[macro_use]
extern crate pest_derive;

mod prelude {
	pub use thiserror::Error;
	pub use eyre::{Result, eyre};
	pub use pest::{
		iterators::{Pair, Pairs},
		pratt_parser::PrattParser,
		Parser,
	};
	pub use std::collections::HashMap;
	pub use std::collections::HashSet;
	pub use crate::parser::*;
	pub use crate::ast::*;
	pub use crate::analyzer::*;
}

use std::fs;
use prelude::*;

fn main() -> Result<()> {
    let unparsed_file = fs::read_to_string("napkins/analysis.napkin")?;
	let pairs = NapkinParser::parse(Rule::file, &unparsed_file)?;
	let napkin = parse_napkin(pairs);
	let _analysis = analyze(&napkin);
	// println!("{:#?}", napkin);
    Ok(())
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
