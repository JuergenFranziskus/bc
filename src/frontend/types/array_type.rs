use super::Type;

pub struct ArrayType {
    id: ArrTypeID,
    member_type: Type,
    length: u64,
}
impl ArrayType {
    pub(super) fn new(id: ArrTypeID, member_type: Type, length: u64) -> Self {
        Self {
            id,
            member_type,
            length,
        }
    }

    pub(super) fn is_equivalent_to(&self, other: &Self) -> bool {
        self.member_type == other.member_type && self.length == other.length
    }

    pub fn id(&self) -> ArrTypeID {
        self.id
    }
    pub fn member(&self) -> Type {
        self.member_type
    }
    pub fn length(&self) -> u64 {
        self.length
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrTypeID(pub usize);
