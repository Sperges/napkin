// use std::{path::PathBuf, fs};

// use clap::{Parser, command};
// use eyre::{Result, Context};
// use rustyline::{Editor, error::ReadlineError};

// pub struct Cell {
//     _raw: String,
//     _computed: f64,
//     _output: String,
// }

// impl Cell {
//     pub fn new(raw: String) -> Self {
//         Self {
//             _raw: raw,
//             _computed: 0.0,
//             _output: "".to_string(),
//         }
//     }
// }

// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
// struct Cli {
//     path: Option<PathBuf>,
//     history: Option<PathBuf>,
// }

// pub struct NapkinCli {
//     cli: Cli,
//     cells: Vec<Cell>,
// }

// impl NapkinCli {
//     pub fn new() -> Self {
//         Self { 
//             cli: Cli::parse() ,
//             cells: Vec::new(),
//         }
//     }

//     pub fn run_cli(&mut self) -> Result<()> {
//         match &self.cli.path {
//             Some(path) => self.run_file(&path),
//             None => self.run_prompt(),
//         }
//     }

//     fn run_file(&self, path: &PathBuf) -> Result<()> {
//         let data = fs::read_to_string(path)?;
//         self.run(&data)?;
//         Ok(())
//     }

//     fn run_prompt(&mut self) -> Result<()> {
//         println!("No input file detected... Starting Napkin prompt.");
//         let mut rl = Editor::<()>::new()?;
//         let history = match &self.cli.history {
//             Some(history) => history.clone(),
//             None => PathBuf::from("napkin.history"),
//         };
//         if rl.load_history(&history).is_err() {
//             println!("No previous history... creating napkin.history file.");
//         }
//         loop {
//             let line = rl.readline("> ");
//             match line {
//                 Ok(line) => {
//                     match self.run_line(line.to_string()) {
//                         Ok(_) => continue,
//                         Err(err) => println!("{}", err),
//                     }
//                 }
//                 Err(ReadlineError::Interrupted) => {
//                     println!("CTRL-C");
//                     break;
//                 }
//                 Err(ReadlineError::Eof) => {
//                     println!("CTRL-D");
//                     break;
//                 }
//                 Err(err) => {
//                     println!("Error: {:?}", err);
//                     break;
//                 }
//             }
//         }
//         rl.save_history(&history)
//             .wrap_err_with(|| "something went wrong when saving history")
//     }

//     fn run_line(&mut self, line: String) -> Result<()> {
//         self.cells.push(Cell::new(line));
//         Ok(())
//     }

//     fn run(&self, _data: &str) -> Result<()> {
//         todo!()
//     }
// }

// fn main() -> Result<()> {
//     NapkinCli::new().run_cli()
// }
