use bc::frontend::{lexer::lex, parser::Parser, type_checker::TypeChecker};

fn main() {
    let src = std::fs::read_to_string("test_programs/mandelbrot.bc").unwrap();
    let tokens = lex(&src);

    let ast = Parser::new(&tokens).parse();
    let (expr_tree, _, _) = TypeChecker::new().type_check(&ast);
    println!("{expr_tree:#?}");
}
