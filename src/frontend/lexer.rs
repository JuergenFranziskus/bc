use super::span::Span;
use logos::Logos;

pub fn lex(src: &str) -> Vec<Token> {
    let lexer = TokenKind::lexer(src).spanned();
    lexer
        .map(|(k, s)| {
            let span = Span::from_points(s.start, s.end);
            Token { span, kind: k }
        })
        .collect()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Token<'a> {
    pub span: Span,
    pub kind: TokenKind<'a>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Logos)]
pub enum TokenKind<'a> {
    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
    #[token("[")]
    OpenSquare,
    #[token("]")]
    CloseSquare,
    #[token("{")]
    OpenCurly,
    #[token("}")]
    CloseCurly,

    #[token("=")]
    Equal,
    #[token("==")]
    DoubleEqual,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("<<")]
    DoubleLess,
    #[token(">>")]
    DoubleGreater,

    #[token("<")]
    LessThan,
    #[token(">")]
    GreaterThan,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("~")]
    Tilde,
    #[token("!")]
    Exclamation,

    #[token("&")]
    Ampersand,
    #[token("&&")]
    DoubleAmpersand,
    #[token("|")]
    Pipe,
    #[token("||")]
    DoublePipe,
    #[token("^")]
    Caret,

    #[token("::")]
    DoubleColon,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,

    #[token("=>")]
    DoubleRightArrow,

    #[token("let")]
    LetKeyword,
    #[token("mut")]
    MutKeyword,
    #[token("while")]
    WhileKeyword,
    #[token("if")]
    IfKeyword,
    #[token("else")]
    ElseKeyword,
    #[token("true")]
    TrueKeyword,
    #[token("false")]
    FalseKeyword,
    #[token("make_slice")]
    MakeSliceKeyword,
    #[token("slen")]
    SLenKeyword,
    #[token("as")]
    AsKeyword,
    #[token("cast")]
    CastKeyword,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier(&'a str),
    #[regex(r"-?[0-9][0-9_]*(?:u8|i8|u16|i16|u32|i32|u64|i64)?")]
    Decimal(&'a str),

    #[error]
    #[regex(r"[\n\t ]+", logos::skip)]
    Error,
}
