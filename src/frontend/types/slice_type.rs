use super::Type;

pub struct SliceType {
    id: SliceTypeID,
    member: Type,
}
impl SliceType {
    pub(super) fn new(id: SliceTypeID, member: Type) -> Self {
        Self { id, member }
    }

    pub(super) fn is_equivalent_to(&self, other: &Self) -> bool {
        self.member == other.member
    }

    pub fn id(&self) -> SliceTypeID {
        self.id
    }
    pub fn member(&self) -> Type {
        self.member
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceTypeID(pub usize);
