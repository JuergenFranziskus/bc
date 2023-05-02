use super::Type;

pub struct FunctionType {
    id: FuncTypeID,
    return_type: Type,
    parameter_types: Vec<Type>,
}
impl FunctionType {
    pub(super) fn new(id: FuncTypeID, return_type: Type, parameter_types: Vec<Type>) -> Self {
        Self {
            id,
            return_type,
            parameter_types,
        }
    }

    pub(super) fn is_equivalent_to(&self, other: &Self) -> bool {
        self.return_type == other.return_type && self.parameter_types == other.parameter_types
    }

    pub fn id(&self) -> FuncTypeID {
        self.id
    }
    pub fn return_type(&self) -> Type {
        self.return_type
    }
    pub fn parameter_types(&self) -> &[Type] {
        &self.parameter_types
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncTypeID(pub usize);
