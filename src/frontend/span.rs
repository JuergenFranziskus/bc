#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub length: usize,
}
impl Span {
    pub fn new(start: usize, length: usize) -> Span {
        Self { start, length }
    }
    pub fn from_points(start: usize, end: usize) -> Span {
        debug_assert!(start <= end);
        let length = end - start;
        Self::new(start, length)
    }
    pub fn merge(self, other: Self) -> Self {
        let start = self.start.min(other.start);
        let end = self.end().max(other.end());
        Self::from_points(start, end)
    }

    pub fn end(self) -> usize {
        self.start + self.length
    }
}
