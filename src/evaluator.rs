// pub fn eval_cells(exprs: &HashMap<String, Expr>) -> Vec<Result<f64>> {
// 	let results = Vec::new();
// 	for expr in exprs {

// 	}
// 	results
// }

// pub fn eval_expr(exprs: &HashMap<String, Expr>, expr: &Expr) -> Result<f64> {
// 	match expr {
// 		Expr::Ident(ident) => eval_ident(exprs, ident),
// 		Expr::Number(number) => Ok(*number),
// 		Expr::Unary {op, expr} => eval_unary(exprs, op, expr),
// 		Expr::Binary {lhs, op, rhs} => eval_binary(exprs, lhs, op, rhs),
// 		Expr::Func(ident, args) => eval_func(exprs, ident, args),
// 	}
// }

// fn eval_ident(exprs: &HashMap<String, Expr>, ident: &String) -> Result<f64> {
// 	todo!()
// }

// fn eval_func(exprs: &HashMap<String, Expr>, ident: &String, args: &Vec<Expr>) -> Result<f64> {
// 	todo!()
// }

// fn eval_unary(exprs: &HashMap<String, Expr>, op: &UnaryOp, expr: &Expr) -> Result<f64> {
// 	let expr = eval_expr(exprs, expr)?;
// 	Ok(match op {
// 		UnaryOp::Negative => -expr,
// 		UnaryOp::Factorial => ((1..(expr as usize)+1).product::<usize>()) as f64,
// 	})
// }

// fn eval_binary(exprs: &HashMap<String, Expr>, lhs: &Expr, op: &BinaryOp, rhs: &Expr) -> Result<f64> {
// 	let lhs = eval_expr(exprs, lhs)?;
// 	let rhs = eval_expr(exprs, rhs)?;
// 	Ok(match op {
// 		BinaryOp::Add => lhs + rhs,
// 		BinaryOp::Subtract => lhs - rhs,
// 		BinaryOp::Multiply => lhs * rhs,
// 		BinaryOp::Divide => lhs / rhs,
// 		BinaryOp::Power => lhs.powf(rhs),
// 		BinaryOp::Modulo => lhs % rhs,
// 	})
// }