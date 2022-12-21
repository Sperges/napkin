// pub fn create_ref_table(cells: &HashMap<String, Expr>) -> HashMap<String, Vec<String>> {
// 	let mut ref_table: HashMap<String, Vec<String>> = HashMap::new();
// 	for key in cells.keys() {
// 		ref_table.insert(key.to_string(), get_cell_refs(cells, key));
// 	}
// 	ref_table
// }

// pub fn get_cell_refs(cells: &HashMap<String, Expr>, key: &String) -> Vec<String> {
// 	let mut refs = Vec::new();
// 	if let Some(expr) = cells.get(key) {
// 		refs.append(&mut match expr {
// 			Expr::Func(key, _) => Vec::new(),
// 			Expr::Ident(_) => Vec::new(),
// 			Expr::Number(_) => Vec::new(),
// 			Expr::Unary { op, expr } => Vec::new(),
// 			Expr::Binary { lhs, op, rhs } => Vec::new(),
// 		});

// 	}
// 	refs
// }

use crate::prelude::*;

pub fn analyze(napkin_file: &Napkin) -> HashMap<String, Result<()>> {
    let analysis = HashMap::new();
    let ref_table = build_ref_table(napkin_file);
	println!("Ref table: {:#?}", ref_table);
    analysis
}

fn build_ref_table(napkin_file: &Napkin) -> HashMap<String, HashSet<String>> {
    let mut ref_table = HashMap::new();
    for group in napkin_file.iter() {
        match group {
            Group::Text(_) => continue,
            Group::Cell(cell) => {
                match cell.as_ref() {
                    Cell::Standard { ident, expr } => {
                        let refs = get_expr_refs(expr);
                        ref_table.insert(ident.to_string(), refs);
                    }
                    Cell::Functional { ident, args, expr } => {
                        let refs = get_expr_refs_with_args(get_args_refs(args), expr);
                        ref_table.insert(ident.to_string(), refs);
                    }
                };
            }
        };
    }
    ref_table
}

fn get_args_refs(args: &Vec<String>) -> HashSet<String> {
    args.into_iter().map(|arg| arg.to_string()).collect()
}

fn get_expr_refs(expr: &Expr) -> HashSet<String> {
    match expr {
        Expr::Func(ident, args) => {
            let mut refs = ident_as_hashset(ident);
            for arg in args.iter() {
                let arg_refs = get_expr_refs(arg.as_ref());
                refs.extend(arg_refs);
            }
            refs
        }
        Expr::Ident(ident) => ident_as_hashset(ident),
        Expr::Number(_) => HashSet::new(),
        Expr::Unary { op: _, expr } => get_expr_refs(expr.as_ref()),
        Expr::Binary { lhs, op: _, rhs } => {
            let mut refs = get_expr_refs(lhs.as_ref());
            refs.extend(get_expr_refs(rhs.as_ref()));
            refs
        }
    }
}

fn get_expr_refs_with_args(args: HashSet<String>, expr: &Expr) -> HashSet<String> {
    get_expr_refs(expr)
        .difference(&args)
        .map(|reference| reference.to_string())
        .collect()
}

fn ident_as_hashset(ident: &String) -> HashSet<String> {
    let mut refs = HashSet::new();
    refs.insert(ident.to_string());
    refs
}
