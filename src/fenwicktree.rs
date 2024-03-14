use std::{
    cmp::{max, min},
    iter::successors,
    marker::PhantomData,
    ops::{Add, Mul},
};

#[inline]
fn lsb(n: usize) -> usize {
    n & n.wrapping_neg()
}

// TODO: I just copied *Identity from segtree.rs
//       Create another file to define these traits?

/// A helper trait for type which has identity of additive.
trait AdditiveIdentity: Add<Self, Output = Self> + Sized {
    fn identity() -> Self;
}

/// A helper trait for type which has identity of multiplicative.
trait MultiplicativeIdentity: Mul<Self, Output = Self> + Sized {
    fn identity() -> Self;
}

/// A helper trait for type which has identity of max.
trait MaxIdentity: Ord {
    fn identity() -> Self;
}

/// A helper trait for type which has identity of min.
trait MinIdentity: Ord {
    fn identity() -> Self;
}

macro_rules! impl_int {
    ($($type:ident),*) => {$(
        impl AdditiveIdentity for $type {
            fn identity() -> Self { 0 }
        }

        impl MultiplicativeIdentity for $type {
            fn identity() -> Self { 1 }
        }

        impl MaxIdentity for $type {
            fn identity() -> Self { $type::MIN }
        }

        impl MinIdentity for $type {
            fn identity() -> Self { $type::MAX }
        }
    )*};
}

macro_rules! impl_float {
    ($($type:ident),*) => {$(
        impl AdditiveIdentity for $type {
            fn identity() -> Self { 0.0 }
        }

        impl MultiplicativeIdentity for $type {
            fn identity() -> Self { 1.0 }
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

pub trait CommutativeMonoid {
    type S: Clone;
    fn identity() -> Self::S;
    fn operate(a: &Self::S, b: &Self::S) -> Self::S;
}

pub struct Additive<T>(PhantomData<fn() -> T>);

impl<T: AdditiveIdentity + Clone> CommutativeMonoid for Additive<T> {
    type S = T;
    fn identity() -> Self::S {
        T::identity()
    }
    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        a.clone() + b.clone()
    }
}

pub struct Multiplicative<T>(PhantomData<fn() -> T>);

impl<T: MultiplicativeIdentity + Clone> CommutativeMonoid for Multiplicative<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        a.clone() * b.clone()
    }
}

pub struct Max<T>(PhantomData<fn() -> T>);

impl<T: MaxIdentity + Clone> CommutativeMonoid for Max<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        max(a, b).clone()
    }
}

pub struct Min<T>(PhantomData<fn() -> T>);

impl<T: MinIdentity + Clone> CommutativeMonoid for Min<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        min(a, b).clone()
    }
}

pub struct FenwickTree<T: CommutativeMonoid> {
    node: Vec<T::S>,
    n: usize,
}

impl<T: CommutativeMonoid> From<Vec<T::S>> for FenwickTree<T> {
    /// TODO: O(nlogn) => O(n)?
    fn from(value: Vec<T::S>) -> Self {
        let mut res = Self::new(value.len());
        for (p, value) in value.into_iter().enumerate() {
            res.update(p, value);
        }
        res
    }
}

impl<T: CommutativeMonoid> From<&[T::S]> for FenwickTree<T> {
    fn from(value: &[T::S]) -> Self {
        Self::from(value.to_vec())
    }
}

impl<T: CommutativeMonoid> From<&Vec<T::S>> for FenwickTree<T> {
    fn from(value: &Vec<T::S>) -> Self {
        Self::from(value.clone())
    }
}

impl<T: CommutativeMonoid> FenwickTree<T> {
    pub fn new(n: usize) -> Self {
        Self {
            node: vec![T::identity(); n + 1],
            n,
        }
    }

    /// a[p] = operate(a[p], value)
    pub fn update(&mut self, p: usize, value: T::S) {
        let p = p + 1; // 0-index to 1-index
        let ps = successors(Some(p), |&p| Some(p + lsb(p))).take_while(|&p| p <= self.n);
        for p in ps {
            self.node[p] = T::operate(&self.node[p], &value);
        }
    }

    /// operate(a[0], a[1], ..., a[p-1])
    pub fn prefix(&mut self, p: usize) -> T::S {
        let ps = successors(Some(p), |&p| Some(p - lsb(p))).take_while(|&p| p > 0);
        let mut res = T::identity();
        for p in ps {
            res = T::operate(&res, &self.node[p]);
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::{Additive, FenwickTree, Max, Min, Multiplicative};

    #[test]
    fn test_additive() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::<Additive<usize>>::from(&a);
        for p in 0..=a.len() {
            assert_eq!(ft.prefix(p), a[0..p].iter().sum());
        }
        a[3] += 20;
        ft.update(3, 20);
        for p in 0..=a.len() {
            assert_eq!(ft.prefix(p), a[0..p].iter().sum());
        }
    }

    #[test]
    fn test_multiplicative() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::<Multiplicative<usize>>::from(&a);
        for p in 0..=a.len() {
            assert_eq!(ft.prefix(p), a[0..p].iter().product());
        }
        a[3] *= 20;
        ft.update(3, 20);
        for p in 0..=a.len() {
            assert_eq!(ft.prefix(p), a[0..p].iter().product());
        }
    }

    #[test]
    fn test_max() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::<Max<usize>>::from(&a);
        for p in 0..=a.len() {
            assert_eq!(
                ft.prefix(p),
                a[0..p].iter().copied().max().unwrap_or(usize::MIN)
            );
        }
        a[3] = a[3].max(20);
        ft.update(3, 20);
        for p in 0..=a.len() {
            assert_eq!(
                ft.prefix(p),
                a[0..p].iter().copied().max().unwrap_or(usize::MIN)
            );
        }
    }

    #[test]
    fn test_min() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::<Min<usize>>::from(&a);
        for p in 0..=a.len() {
            assert_eq!(
                ft.prefix(p),
                a[0..p].iter().copied().min().unwrap_or(usize::MAX)
            );
        }
        a[3] = a[3].min(20);
        ft.update(3, 20);
        for p in 0..=a.len() {
            assert_eq!(
                ft.prefix(p),
                a[0..p].iter().copied().min().unwrap_or(usize::MAX)
            );
        }
    }
}
