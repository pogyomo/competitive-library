use std::{
    collections::{btree_map::Range as BTreeMapRange, BTreeMap},
    fmt::Debug,
    ops::RangeBounds,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Multiset<T> {
    kinds: usize,
    size: usize,
    map: BTreeMap<T, usize>,
}

impl<T: Debug + Ord> Debug for Multiset<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.range(..)).finish()
    }
}

impl<T> Multiset<T> {
    pub fn new() -> Self {
        Self {
            kinds: 0,
            size: 0,
            map: BTreeMap::new(),
        }
    }
}

impl<T: Ord> Multiset<T> {
    pub fn insert(&mut self, value: T) {
        self.size += 1;
        if let Some(count) = self.map.get(&value) {
            self.map.insert(value, count + 1);
        } else {
            self.kinds += 1;
            self.map.insert(value, 1);
        }
    }

    pub fn remove(&mut self, value: &T) {
        self.size = self.size.saturating_sub(1);
        match self.map.get_mut(value) {
            Some(count) if *count >= 2 => *count -= 1,
            Some(_) => {
                self.kinds -= 1;
                self.map.remove(value);
            }
            _ => (),
        }
    }

    pub fn count(&self, value: &T) -> usize {
        self.map.get(value).copied().unwrap_or(0)
    }

    pub fn range<R: RangeBounds<T>>(&self, range: R) -> Range<'_, T> {
        Range::new(self.map.range(range))
    }
}

impl<T> Multiset<T> {
    /// Returns the number of different elements in the set.
    pub fn kinds(&self) -> usize {
        self.kinds
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.size
    }
}

pub struct Range<'a, T> {
    front_count: usize,
    front: Option<&'a T>,
    back_count: usize,
    back: Option<&'a T>,
    iter: BTreeMapRange<'a, T, usize>,
}

// Clone can be implemented to Range even if T doesn't implement Clone.
impl<T> Clone for Range<'_, T> {
    fn clone(&self) -> Self {
        Range {
            front_count: self.front_count,
            front: self.front,
            back_count: self.back_count,
            back: self.back,
            iter: self.iter.clone(),
        }
    }
}

impl<T: Debug> Debug for Range<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<'a, T> Range<'a, T> {
    pub fn new(iter: BTreeMapRange<'a, T, usize>) -> Self {
        Self {
            front_count: 0,
            front: None,
            back_count: 0,
            back: None,
            iter,
        }
    }
}

impl<'a, T> Iterator for Range<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_count == 0 {
            if let Some((value, count)) = self.iter.next() {
                self.front = Some(value);
                self.front_count = *count;
            } else {
                if self.back_count != 0 {
                    self.front = self.back;
                    self.front_count = 1;
                    self.back_count -= 1;
                } else {
                    self.front = None;
                    self.front_count = usize::MAX;
                }
            }
        }
        self.front_count = self.front_count.saturating_sub(1);
        self.front
    }
}

impl<'a, T> DoubleEndedIterator for Range<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.back_count == 0 {
            if let Some((value, count)) = self.iter.next_back() {
                self.back = Some(value);
                self.back_count = *count;
            } else {
                if self.front_count != 0 {
                    self.back = self.front;
                    self.back_count = 1;
                    self.front_count -= 1;
                } else {
                    self.back = None;
                    self.back_count = usize::MAX;
                }
            }
        }
        self.back_count = self.back_count.saturating_sub(1);
        self.back
    }
}

#[cfg(test)]
mod test {
    use super::Multiset;

    #[test]
    fn get_only_from_front() {
        let mut ms = Multiset::new();
        ms.insert(1);
        ms.insert(1);
        ms.insert(2);
        ms.insert(3);
        let mut iter = ms.range(..);
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
        let mut iter = ms.range(2..);
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn get_only_from_back() {
        let mut ms = Multiset::new();
        ms.insert(1);
        ms.insert(1);
        ms.insert(2);
        ms.insert(3);
        let mut iter = ms.range(..);
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next_back(), Some(&2));
        assert_eq!(iter.next_back(), Some(&1));
        assert_eq!(iter.next_back(), Some(&1));
        assert_eq!(iter.next_back(), None);
        let mut iter = ms.range(2..);
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next_back(), Some(&2));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn get_from_both_side() {
        let mut ms = Multiset::new();
        ms.insert(1);
        ms.insert(1);
        ms.insert(2);
        ms.insert(3);
        let mut iter = ms.range(..);
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next_back(), Some(&2));
        assert_eq!(iter.next_back(), Some(&1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        let mut iter = ms.range(2..);
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn insert_and_remove() {
        let mut ms = Multiset::new();
        ms.insert(1);
        ms.insert(1);
        ms.insert(2);
        ms.insert(3);
        assert_eq!(ms.range(..).collect::<Vec<_>>(), vec![&1, &1, &2, &3]);
        ms.remove(&1);
        assert_eq!(ms.range(..).collect::<Vec<_>>(), vec![&1, &2, &3]);
        ms.remove(&2);
        assert_eq!(ms.range(..).collect::<Vec<_>>(), vec![&1, &3]);
        ms.remove(&3);
        assert_eq!(ms.range(..).collect::<Vec<_>>(), vec![&1]);
        ms.remove(&1);
        assert_eq!(ms.range(..).collect::<Vec<_>>(), Vec::<&usize>::new());
    }

    #[test]
    fn test_kinds() {
        let mut ms = Multiset::new();
        ms.insert(1);
        ms.insert(2);
        ms.insert(3);
        assert_eq!(ms.kinds(), 3);
        ms.insert(3);
        assert_eq!(ms.kinds(), 3);
        ms.remove(&3);
        assert_eq!(ms.kinds(), 3);
        ms.remove(&3);
        assert_eq!(ms.kinds(), 2);
    }
}
