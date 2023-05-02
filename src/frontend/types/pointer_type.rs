use super::Type;

pub struct PointerType {
    id: PointerTypeID,
    mutable: bool,
    pointee: Type,
}
impl PointerType {
    pub(super) fn new(id: PointerTypeID, mutable: bool, pointee: Type) -> Self {
        Self {
            id,
            mutable,
            pointee,
        }
    }

    pub(super) fn is_equivalent_to(&self, other: &Self) -> bool {
        self.mutable == other.mutable && self.pointee == other.pointee
    }

    pub fn id(&self) -> PointerTypeID {
        self.id
    }
    pub fn mutable(&self) -> bool {
        self.mutable
    }
    pub fn pointee(&self) -> Type {
        self.pointee
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointerTypeID(pub usize);
