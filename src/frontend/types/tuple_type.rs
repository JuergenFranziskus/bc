use super::Type;

pub struct TupleType {
    id: TupleTypeID,
    members: Vec<Type>,
}
impl TupleType {
    pub(super) fn new(id: TupleTypeID, members: Vec<Type>) -> Self {
        Self { id, members }
    }
    pub(super) fn is_equivalent_to(&self, other: &Self) -> bool {
        self.members == other.members
    }

    pub fn id(&self) -> TupleTypeID {
        self.id
    }
    pub fn members(&self) -> &[Type] {
        &self.members
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TupleTypeID(pub usize);
