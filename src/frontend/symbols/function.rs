use super::{variable::VarID, Symbols};
use crate::frontend::types::{function_type::FuncTypeID, Type, Types};
use std::collections::HashSet;

pub struct Function<'a> {
    id: FunctionID,
    name: &'a str,
    ret_type: Type,
    parameters: Vec<VarID>,
    variables: HashSet<VarID>,
}
impl<'a> Function<'a> {
    pub(super) fn new(id: usize, name: &'a str, ret_type: Type) -> Self {
        Self {
            id: FunctionID(id),
            name,
            ret_type,
            parameters: Vec::new(),
            variables: HashSet::new(),
        }
    }

    pub(super) fn add_var(&mut self, var: VarID) {
        self.variables.insert(var);
    }
    pub(super) fn add_parameter(&mut self, var: VarID) {
        self.parameters.push(var);
    }

    pub fn id(&self) -> FunctionID {
        self.id
    }
    pub fn name(&self) -> &'a str {
        self.name
    }
    pub fn return_type(&self) -> Type {
        self.ret_type
    }
    pub fn parameters(&self) -> &[VarID] {
        &self.parameters
    }
    pub fn variables(&self) -> &HashSet<VarID> {
        &self.variables
    }

    pub fn function_type(&self, symbols: &Symbols, types: &mut Types) -> FuncTypeID {
        let ret_type = self.ret_type;
        let param_types = self
            .parameters
            .iter()
            .map(|&v| symbols[v].var_type())
            .collect();

        types.add_func_type(ret_type, param_types)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionID(pub usize);
