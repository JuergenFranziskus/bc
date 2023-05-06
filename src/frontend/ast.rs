use super::span::Span;
use crate::frontend::types::IntType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ast<'a> {
    pub functions: Vec<Function<'a>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function<'a> {
    pub span: Span,
    pub name: &'a str,
    pub return_type: Option<TypeExpr<'a>>,
    pub parameters: Vec<FunctionParameter<'a>>,
    pub body: Expr<'a>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionParameter<'a> {
    pub span: Span,
    pub name: &'a str,
    pub param_type: TypeExpr<'a>,
    pub mutable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr<'a> {
    pub span: Span,
    pub kind: ExprKind<'a>,
}
impl Expr<'_> {
    pub fn is_structural(&self) -> bool {
        match &self.kind {
            ExprKind::Block(_, _) => true,
            ExprKind::If(_, _, _) => true,
            ExprKind::While(_, _) => true,
            _ => false,
        }
    }

    pub fn is_int_literal(&self) -> bool {
        matches!(self.kind, ExprKind::Decimal(_, _))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind<'a> {
    Paren(Box<Expr<'a>>),

    MakeSlice(Box<Expr<'a>>, Box<Expr<'a>>),
    SliceLength(Box<Expr<'a>>),

    TupleIndex(Box<Expr<'a>>, u64),
    Index(Box<Expr<'a>>, Box<Expr<'a>>),
    Call(Box<Expr<'a>>, Vec<Expr<'a>>),
    Intrinsic(Intrinsic, Vec<Expr<'a>>),
    Declaration(&'a str, bool, Option<Box<TypeExpr<'a>>>, Box<Expr<'a>>),

    BitCast(Box<Expr<'a>>, Box<TypeExpr<'a>>),
    Cast(Box<Expr<'a>>, Box<TypeExpr<'a>>),

    If(Box<Expr<'a>>, Box<Expr<'a>>, Option<Box<Expr<'a>>>),
    While(Box<Expr<'a>>, Box<Expr<'a>>),
    Block(Vec<Expr<'a>>, Option<Box<Expr<'a>>>),
    Binary(BinaryOp, Box<Expr<'a>>, Box<Expr<'a>>),
    Prefix(PrefixOp, Box<Expr<'a>>),

    Tuple(Vec<Expr<'a>>),
    Array(Vec<Expr<'a>>),
    ShortArray(Box<Expr<'a>>, Box<Expr<'a>>),
    Named(&'a str),
    Decimal(i128, Option<IntSuffix>),
    Bool(bool),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntSuffix {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    USize,
    ISize,
}
impl IntSuffix {
    pub fn to_type(self) -> IntType {
        use IntSuffix::*;
        match self {
            U8 => IntType::u8(),
            I8 => IntType::i8(),
            U16 => IntType::u16(),
            I16 => IntType::i16(),
            U32 => IntType::u32(),
            I32 => IntType::i32(),
            U64 => IntType::u64(),
            I64 => IntType::i64(),
            USize => IntType::usize(),
            ISize => IntType::isize(),
        }
    }

    pub fn parse_from_src(src: &str) -> Option<IntSuffix> {
        let get_double_suffix = || {
            if src.len() < 2 {
                return None;
            }
            let last_two = &src[(src.len() - 2)..];
            match last_two {
                "u8" => Some(IntSuffix::U8),
                "i8" => Some(IntSuffix::I8),
                _ => None,
            }
        };
        let get_triple_suffix = || {
            if src.len() < 3 {
                return None;
            }
            let last_three = &src[(src.len() - 3)..];
            match last_three {
                "u16" => Some(IntSuffix::U16),
                "i16" => Some(IntSuffix::I16),
                "u32" => Some(IntSuffix::U32),
                "i32" => Some(IntSuffix::I32),
                "u64" => Some(IntSuffix::U64),
                "i64" => Some(IntSuffix::I64),
                _ => None,
            }
        };
        let get_quintuple_suffix = || {
            if src.len() < 5 {
                return None;
            }
            let last_five = &src[(src.len() - 5)..];
            match last_five {
                "usize" => Some(IntSuffix::USize),
                "isize" => Some(IntSuffix::ISize),
                _ => None,
            }
        };

        get_quintuple_suffix()
            .or_else(get_triple_suffix)
            .or_else(get_double_suffix)
    }
    pub fn src_len(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 2,
            Self::USize | Self::ISize => 5,
            _ => 3,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    ShiftLeft,
    ShiftRight,

    TestEqual,
    TestNotEqual,
    TestGreater,
    TestLess,
    TestGreaterEqual,
    TestLessEqual,

    BitAnd,
    BitOr,
    BitXor,
    And,
    Or,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrefixOp {
    BitNot,
    Not,
    Negate,
    AddrOf,
    AddrOfMut,
    Deref,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    VolatileStore,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeExpr<'a> {
    pub span: Span,
    pub kind: TypeExprKind<'a>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeExprKind<'a> {
    Pointer(bool, Box<TypeExpr<'a>>),
    Array(Expr<'a>, Box<TypeExpr<'a>>),
    Slice(Box<TypeExpr<'a>>),
    Tuple(Vec<TypeExpr<'a>>),
    Named(&'a str),
}
