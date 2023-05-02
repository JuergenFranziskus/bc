use self::{
    function::{Function, FunctionID},
    variable::{VarID, Variable},
};
use super::types::Type;
use std::ops::{Index, IndexMut};

pub mod function;
pub mod value_kind;
pub mod variable;

pub struct Symbols<'a> {
    functions: Vec<Function<'a>>,
    variables: Vec<Variable<'a>>,
}
impl<'a> Symbols<'a> {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            variables: Vec::new(),
        }
    }

    pub fn add_function(&mut self, name: &'a str, ret_type: impl Into<Type>) -> FunctionID {
        let id = self.functions.len();
        self.functions
            .push(Function::new(id, name, ret_type.into()));
        FunctionID(id)
    }
    pub fn add_var(
        &mut self,
        fid: FunctionID,
        name: &'a str,
        var_type: impl Into<Type>,
        mutable: bool,
    ) -> VarID {
        let id = VarID(self.variables.len());
        let var = Variable::new(id, fid, name, var_type.into(), mutable.into());
        self.variables.push(var);
        self[fid].add_var(id);
        id
    }
    pub fn add_parameter(
        &mut self,
        fid: FunctionID,
        name: &'a str,
        var_type: impl Into<Type>,
        mutable: bool,
    ) -> VarID {
        let id = self.add_var(fid, name, var_type, mutable);
        self[fid].add_parameter(id);
        id
    }

    pub fn resolve_function(&self, name: &str) -> Option<FunctionID> {
        for function in &self.functions {
            if function.name() == name {
                return Some(function.id());
            }
        }
        None
    }
}
impl<'a> Index<FunctionID> for Symbols<'a> {
    type Output = Function<'a>;

    fn index(&self, index: FunctionID) -> &Self::Output {
        &self.functions[index.0]
    }
}
impl<'a> IndexMut<FunctionID> for Symbols<'a> {
    fn index_mut(&mut self, index: FunctionID) -> &mut Self::Output {
        &mut self.functions[index.0]
    }
}
impl<'a> Index<VarID> for Symbols<'a> {
    type Output = Variable<'a>;

    fn index(&self, index: VarID) -> &Self::Output {
        &self.variables[index.0]
    }
}
