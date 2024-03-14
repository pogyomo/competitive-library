use std::iter::successors;
use std::ops::AddAssign;

macro_rules! impl_int {
    ($($type:ident),*) => {$(
        impl FenwickTreeElement for $type {
            fn identity() -> Self { 0 }
        }
    )*};
}

macro_rules! impl_float {
    ($($type:ident),*) => {$(
        impl FenwickTreeElement for $type {
            fn identity() -> Self { 0.0 }
        }
    )*};
}

impl_int! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize
}

impl_float! {
    f32, f64
}

#[inline]
fn lsb(n: usize) -> usize {
    n & n.wrapping_neg()
}

pub trait FenwickTreeElement: AddAssign<Self> + Clone + Sized {
    fn identity() -> Self;
}

pub struct FenwickTree<T> {
    node: Vec<T>,
    n: usize,
}

impl<T: FenwickTreeElement> From<Vec<T>> for FenwickTree<T> {
    /// TODO: O(nlogn) => O(n)?
    fn from(value: Vec<T>) -> Self {
        let mut res = Self::new(value.len());
        for (p, value) in value.into_iter().enumerate() {
            res.add(p, value);
        }
        res
    }
}

impl<T: FenwickTreeElement> From<&[T]> for FenwickTree<T> {
    fn from(value: &[T]) -> Self {
        Self::from(value.to_vec())
    }
}

impl<T: FenwickTreeElement> From<&Vec<T>> for FenwickTree<T> {
    fn from(value: &Vec<T>) -> Self {
        Self::from(value.clone())
    }
}

impl<T: FenwickTreeElement> FenwickTree<T> {
    pub fn new(n: usize) -> Self {
        Self {
            node: vec![T::identity(); n + 1],
            n,
        }
    }

    /// a[p] += value
    pub fn add(&mut self, p: usize, value: T) {
        let p = p + 1; // 0-index to 1-index
        let ps = successors(Some(p), |&p| Some(p + lsb(p))).take_while(|&p| p <= self.n);
        for p in ps {
            self.node[p] += value.clone();
        }
    }

    /// sum in [0, p)
    pub fn sum(&mut self, p: usize) -> T {
        let ps = successors(Some(p), |&p| Some(p - lsb(p))).take_while(|&p| p > 0);
        let mut res = T::identity();
        for p in ps {
            res += self.node[p].clone();
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::FenwickTree;

    #[test]
    fn test() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::<usize>::from(&a);
        for i in 0..=a.len() {
            assert_eq!(ft.sum(i), a[0..i].iter().sum::<usize>());
        }
        a[2] += 10;
        ft.add(2, 10);
        for i in 0..=a.len() {
            assert_eq!(ft.sum(i), a[0..i].iter().sum::<usize>());
        }
    }
}
