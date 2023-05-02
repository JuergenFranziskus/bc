#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueKind {
    RValue,
    LValue(bool),
}
