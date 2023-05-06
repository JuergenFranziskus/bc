use super::{
    symbols::{function::FunctionID, value_kind::ValueKind, variable::VarID},
    types::Type,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExprTree {
    pub functions: Vec<Function>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionID,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    pub expr_type: Type,
    pub value_kind: ValueKind,
    pub kind: ExprKind,
}
impl Expr {
    pub fn unit() -> Expr {
        Expr {
            expr_type: Type::Unit,
            value_kind: ValueKind::RValue,
            kind: ExprKind::Unit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind {
    Assign(Box<Expr>, Box<Expr>),

    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    ShiftLeft(Box<Expr>, Box<Expr>),
    ShiftRight(Box<Expr>, Box<Expr>),
    TestEqual(Box<Expr>, Box<Expr>),
    TestNotEqual(Box<Expr>, Box<Expr>),
    TestGreater(Box<Expr>, Box<Expr>),
    TestLess(Box<Expr>, Box<Expr>),
    TestGreaterEqual(Box<Expr>, Box<Expr>),
    TestLessEqual(Box<Expr>, Box<Expr>),
    BitAnd(Box<Expr>, Box<Expr>),
    BitOr(Box<Expr>, Box<Expr>),
    BitXor(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    SliceLength(Box<Expr>),
    MakeSlice(Box<Expr>, Box<Expr>),

    Negate(Box<Expr>),
    BitNot(Box<Expr>),
    Not(Box<Expr>),

    AddrOf(Box<Expr>),
    AddrOfMut(Box<Expr>),
    Deref(Box<Expr>),
    VolatileStore(Box<Expr>, Box<Expr>),

    Conversion(Box<Expr>),

    Paren(Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    TupleIndex(Box<Expr>, u64),

    If(Box<Expr>, Box<Expr>, Box<Expr>),
    While(Box<Expr>, Box<Expr>),
    Block(Vec<Expr>, Box<Expr>),

    Tuple(Vec<Expr>),
    Array(Vec<Expr>),
    ShortArray(Box<Expr>, u64),
    Variable(VarID),
    Function(FunctionID),
    Decimal(i128),
    Boolean(bool),
    Unit,
}
