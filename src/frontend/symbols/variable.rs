use super::function::FunctionID;
use crate::frontend::types::Type;

pub struct Variable<'a> {
    id: VarID,
    function: FunctionID,
    name: &'a str,
    var_type: Type,
    mutable: bool,
}
impl<'a> Variable<'a> {
    pub(super) fn new(
        id: VarID,
        fid: FunctionID,
        name: &'a str,
        var_type: Type,
        mutable: bool,
    ) -> Variable {
        Self {
            id,
            function: fid,
            name,
            var_type,
            mutable,
        }
    }

    pub fn id(&self) -> VarID {
        self.id
    }
    pub fn function(&self) -> FunctionID {
        self.function
    }
    pub fn name(&self) -> &'a str {
        self.name
    }
    pub fn var_type(&self) -> Type {
        self.var_type
    }
    pub fn mutable(&self) -> bool {
        self.mutable
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarID(pub usize);
