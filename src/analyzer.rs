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