use self::{
    array_type::{ArrTypeID, ArrayType},
    function_type::{FuncTypeID, FunctionType},
    pointer_type::{PointerType, PointerTypeID},
    slice_type::{SliceType, SliceTypeID},
    tuple_type::{TupleType, TupleTypeID},
};
use std::ops::Index;

pub mod array_type;
pub mod function_type;
pub mod pointer_type;
pub mod slice_type;
pub mod tuple_type;

pub struct Types {
    functions: Vec<FunctionType>,
    arrays: Vec<ArrayType>,
    pointers: Vec<PointerType>,
    slices: Vec<SliceType>,
    tuples: Vec<TupleType>,
}
impl Types {
    pub fn new() -> Types {
        Self {
            functions: Vec::new(),
            arrays: Vec::new(),
            pointers: Vec::new(),
            slices: Vec::new(),
            tuples: Vec::new(),
        }
    }

    pub fn add_func_type(&mut self, ret_type: Type, param_types: Vec<Type>) -> FuncTypeID {
        let pot_id = FuncTypeID(self.functions.len());
        let func_type = FunctionType::new(pot_id, ret_type, param_types);

        if let Some(found) = self
            .functions
            .iter()
            .find(|t| t.is_equivalent_to(&func_type))
        {
            found.id()
        } else {
            self.functions.push(func_type);
            pot_id
        }
    }
    pub fn add_array_type(&mut self, member: Type, length: u64) -> ArrTypeID {
        let pot_id = ArrTypeID(self.arrays.len());
        let arr_type = ArrayType::new(pot_id, member, length);

        if let Some(found) = self.arrays.iter().find(|t| t.is_equivalent_to(&arr_type)) {
            found.id()
        } else {
            self.arrays.push(arr_type);
            pot_id
        }
    }
    pub fn add_pointer_type(&mut self, mutable: bool, pointee: Type) -> PointerTypeID {
        let pot_id = PointerTypeID(self.pointers.len());
        let ptr_type = PointerType::new(pot_id, mutable, pointee);

        if let Some(found) = self.pointers.iter().find(|t| t.is_equivalent_to(&ptr_type)) {
            found.id()
        } else {
            self.pointers.push(ptr_type);
            pot_id
        }
    }
    pub fn add_slice_type(&mut self, member: Type) -> SliceTypeID {
        let pot_id = SliceTypeID(self.slices.len());
        let slice_type = SliceType::new(pot_id, member);

        if let Some(found) = self.slices.iter().find(|t| t.is_equivalent_to(&slice_type)) {
            found.id()
        } else {
            self.slices.push(slice_type);
            pot_id
        }
    }
    pub fn add_tuple_type(&mut self, members: Vec<Type>) -> TupleTypeID {
        let pot_id = TupleTypeID(self.slices.len());
        let tuple_type = TupleType::new(pot_id, members);

        if let Some(found) = self.tuples.iter().find(|t| t.is_equivalent_to(&tuple_type)) {
            found.id()
        } else {
            self.tuples.push(tuple_type);
            pot_id
        }
    }
}
impl Index<FuncTypeID> for Types {
    type Output = FunctionType;

    fn index(&self, index: FuncTypeID) -> &Self::Output {
        &self.functions[index.0]
    }
}
impl Index<ArrTypeID> for Types {
    type Output = ArrayType;

    fn index(&self, index: ArrTypeID) -> &Self::Output {
        &self.arrays[index.0]
    }
}
impl Index<PointerTypeID> for Types {
    type Output = PointerType;

    fn index(&self, index: PointerTypeID) -> &Self::Output {
        &self.pointers[index.0]
    }
}
impl Index<SliceTypeID> for Types {
    type Output = SliceType;

    fn index(&self, index: SliceTypeID) -> &Self::Output {
        &self.slices[index.0]
    }
}
impl Index<TupleTypeID> for Types {
    type Output = TupleType;

    fn index(&self, index: TupleTypeID) -> &Self::Output {
        &self.tuples[index.0]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Void,
    Unit,
    Bool,
    Integer(IntType),
    Literal(i128),
    Function(FuncTypeID),
    Array(ArrTypeID),
    Pointer(PointerTypeID),
    Slice(SliceTypeID),
    Tuple(TupleTypeID),
}
impl Type {
    /// A concrete type is a type that has a defined byte representation.
    /// Examples for non-concrete types are void and the type of an integer literal.
    pub fn is_concrete(self) -> bool {
        !matches!(self, Type::Void | Type::Literal(_))
    }

    /// A sized type is a type whose size is known statically.
    /// All types that are not concrete are also not sized.
    pub fn is_sized(self) -> bool {
        self.is_concrete() && !matches!(self, Self::Slice(_))
    }

    pub fn is_void(self) -> bool {
        matches!(self, Self::Void)
    }

    pub fn is_integer(self) -> bool {
        matches!(self, Self::Integer(_))
    }
    pub fn is_integer_literal(self) -> bool {
        matches!(self, Self::Literal(_))
    }
}
impl From<IntType> for Type {
    fn from(value: IntType) -> Self {
        Self::Integer(value)
    }
}
impl From<FuncTypeID> for Type {
    fn from(value: FuncTypeID) -> Self {
        Self::Function(value)
    }
}
impl From<ArrTypeID> for Type {
    fn from(value: ArrTypeID) -> Self {
        Self::Array(value)
    }
}
impl From<PointerTypeID> for Type {
    fn from(value: PointerTypeID) -> Self {
        Self::Pointer(value)
    }
}
impl From<SliceTypeID> for Type {
    fn from(value: SliceTypeID) -> Self {
        Self::Slice(value)
    }
}
impl From<TupleTypeID> for Type {
    fn from(value: TupleTypeID) -> Self {
        Self::Tuple(value)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntType {
    Regular(RegularIntKind, Signed),
    Special(SpecialIntKind, Signed),
}
impl IntType {
    pub fn signed(self) -> bool {
        match self {
            Self::Regular(_, signed) => signed,
            Self::Special(_, signed) => signed,
        }
    }

    pub fn fits_value(self, value: i128) -> bool {
        match self {
            IntType::Regular(t, s) => t.fits_value(s, value),
            IntType::Special(t, s) => t.fits_value(s, value),
        }
    }

    pub fn u8() -> Self {
        Self::Regular(RegularIntKind::Byte, false)
    }
    pub fn u16() -> Self {
        Self::Regular(RegularIntKind::Short, false)
    }
    pub fn u32() -> Self {
        Self::Regular(RegularIntKind::Int, false)
    }
    pub fn u64() -> Self {
        Self::Regular(RegularIntKind::Long, false)
    }
    pub fn i8() -> Self {
        Self::Regular(RegularIntKind::Byte, true)
    }
    pub fn i16() -> Self {
        Self::Regular(RegularIntKind::Short, true)
    }
    pub fn i32() -> Self {
        Self::Regular(RegularIntKind::Int, true)
    }
    pub fn i64() -> Self {
        Self::Regular(RegularIntKind::Long, true)
    }

    pub fn usize() -> Self {
        Self::Special(SpecialIntKind::Size, false)
    }
    pub fn isize() -> Self {
        Self::Special(SpecialIntKind::Size, true)
    }

    pub fn uptr() -> Self {
        Self::Special(SpecialIntKind::Ptr, false)
    }
    pub fn iptr() -> Self {
        Self::Special(SpecialIntKind::Ptr, true)
    }

    pub fn flipped_sign(self) -> Self {
        match self {
            Self::Regular(kind, sign) => Self::Regular(kind, !sign),
            Self::Special(kind, sign) => Self::Special(kind, !sign),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RegularIntKind {
    Byte,
    Short,
    Int,
    Long,
}
impl RegularIntKind {
    pub fn fits_value(self, signed: bool, value: i128) -> bool {
        use RegularIntKind::*;
        match (self, signed) {
            (Byte, false) => (U8_MIN..U8_MAX).contains(&value),
            (Byte, true) => (I8_MIN..I8_MAX).contains(&value),
            (Short, false) => (U16_MIN..U16_MAX).contains(&value),
            (Short, true) => (I16_MIN..I16_MAX).contains(&value),
            (Int, false) => (U32_MIN..U32_MAX).contains(&value),
            (Int, true) => (I32_MIN..I32_MAX).contains(&value),
            (Long, false) => (U64_MIN..U64_MAX).contains(&value),
            (Long, true) => (I64_MIN..I64_MAX).contains(&value),
        }
    }

    fn numerical_rank(self) -> u16 {
        match self {
            Self::Byte => 100,
            Self::Short => 200,
            Self::Int => 300,
            Self::Long => 400,
        }
    }
}
impl PartialOrd for RegularIntKind {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_rank = self.numerical_rank();
        let other_rank = other.numerical_rank();
        Some(self_rank.cmp(&other_rank))
    }
}
impl Ord for RegularIntKind {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SpecialIntKind {
    Size,
    Ptr,
}
impl SpecialIntKind {
    pub fn fits_value(self, signed: bool, value: i128) -> bool {
        RegularIntKind::Short.fits_value(signed, value)
    }
}

pub type Signed = bool;
pub const U8_MIN: i128 = u8::MIN as i128;
pub const I8_MIN: i128 = i8::MIN as i128;
pub const U16_MIN: i128 = u16::MIN as i128;
pub const I16_MIN: i128 = i16::MIN as i128;
pub const U32_MIN: i128 = u32::MIN as i128;
pub const I32_MIN: i128 = i32::MIN as i128;
pub const U64_MIN: i128 = u64::MIN as i128;
pub const I64_MIN: i128 = i64::MIN as i128;
pub const U8_MAX: i128 = u8::MAX as i128;
pub const I8_MAX: i128 = i8::MAX as i128;
pub const U16_MAX: i128 = u16::MAX as i128;
pub const I16_MAX: i128 = i16::MAX as i128;
pub const U32_MAX: i128 = u32::MAX as i128;
pub const I32_MAX: i128 = i32::MAX as i128;
pub const U64_MAX: i128 = u64::MAX as i128;
pub const I64_MAX: i128 = i64::MAX as i128;
