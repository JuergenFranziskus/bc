use std::io::stdout;

use bc::{
    code_gen::{CodeGen, TargetInfo},
    frontend::{lexer::lex, parser::Parser, type_checker::TypeChecker},
};
use cir::printing::Printer;

fn main() {
    let src = std::fs::read_to_string("test_programs/mandelbrot_risce.bc").unwrap();
    let tokens = lex(&src);

    let ast = Parser::new(&tokens).parse();
    let (expr_tree, symbols, types) = TypeChecker::new().type_check(&ast);
    let (module, mut ctypes) = CodeGen::new(
        &symbols,
        &types,
        TargetInfo {
            size_bits: 64,
            ptr_bits: 64,
        },
    )
    .gen_code(&expr_tree);

    let mut printer = Printer::new(stdout(), &module, &mut ctypes);
    printer.pretty_print().unwrap()
}
