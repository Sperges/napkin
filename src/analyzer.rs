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

use std::ops::Deref;

use crate::prelude::*;

pub fn analyze(napkin_file: &Napkin) -> HashMap<String, Result<()>> {
    let analysis = HashMap::new();
    let ref_tree = build_ref_tree(napkin_file);
    // let _results = ref_table.check_refs();
	println!("Ref table: {:#?}", ref_tree);
    analysis
}

#[derive(Debug)]
enum ReferenceTree {
    Terminal,
    Children(Vec<Box<ReferenceTree>>),
}

// #[derive(Debug)]
// struct RefTable {
//     table: HashMap<String, HashSet<String>>
// }

// impl Deref for RefTable {
//     type Target = HashMap<String, HashSet<String>>;

//     fn deref(&self) -> &Self::Target {
//         &self.table
//     }
// }

// impl RefTable {
//     pub fn new(table: HashMap<String, HashSet<String>>) -> Self {
//         Self {
//             table,
//         }
//     }

//     pub fn check_refs(&self) -> HashMap<String, Result<()>> {
//         let results = HashMap::new();
//         for (cell_name, refs) in self.table.iter() {
//             results.insert(cell_name.clone(), )
//         }
//         results
//     }

//     fn check_ref(&self, ref_name: &String, prev_refs: &mut HashSet<&String>) -> Result<()> {

//         Ok(())
//     }
// }

// fn check_ref_table(ref_table: &HashMap<String, HashSet<String>>) -> HashMap<&String, Result<()>> {
//     let mut results = HashMap::new();
//     for (cell_name, refs) in ref_table.iter() {
//         let prev_refs: HashSet<&String> = HashSet::new();
//         for ref_name in refs.iter() {
//             let res = check_ref(ref_table, ref_name, &prev_refs);
//             results.insert(cell_name, res);
//         }
//     }
//     results
// }

// fn check_ref(ref_table: &HashMap<String, HashSet<String>>, ref_name: &String, prev_refs: &HashSet<&String>) -> Result<()> {
//     if prev_refs.contains(ref_name) {
//         todo!("Err, circular ref")
//     }
//     Ok(())
// }



fn build_ref_tree(napkin_file: &Napkin) -> ReferenceTree {
    let mut references = HashMap::new();
    for group in napkin_file.iter() {
        match group {
            Group::Text(_) => continue,
            Group::Cell(cell) => {
                match cell.as_ref() {
                    Cell::Standard { ident, expr } => {
                        let refs = get_expr_refs(expr);
                        references.insert(ident.to_string(), refs);
                    }
                    Cell::Functional { ident, args, expr } => {
                        let refs = get_expr_refs_with_args(get_args_refs(args), expr);
                        references.insert(ident.to_string(), refs);
                    }
                };
            }
        };
    }
    check_refs(&references)
}

fn check_refs(references: &HashMap<String, HashSet<String>>) -> ReferenceTree {
    let references = Vec::new();
    todo!();
    ReferenceTree::Children(references)
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
