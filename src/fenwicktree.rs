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

impl<T: CommutativeMonoid> FromIterator<T::S> for FenwickTree<T> {
    fn from_iter<I: IntoIterator<Item = T::S>>(iter: I) -> Self {
        let mut node = vec![T::identity()];
        node.extend(iter);
        let n = node.len() - 1;
        for i in 1..=n {
            let j = i + lsb(i);
            if j <= n {
                node[j] = T::operate(&node[j], &node[i]);
            }
        }
        Self { node, n }
    }
}

impl<T: CommutativeMonoid> From<Vec<T::S>> for FenwickTree<T> {
    fn from(value: Vec<T::S>) -> Self {
        Self::from_iter(value)
    }
}

impl<T: CommutativeMonoid> FenwickTree<T> {
    /// Construct a new `FenwickTree` with size `n`.
    pub fn new(n: usize) -> Self {
        Self {
            node: vec![T::identity(); n + 1],
            n,
        }
    }

    /// Returns a size of `FenwickTree`.
    ///
    /// Time complexity is O(1).
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if this length is 0.
    ///
    /// Time complexity is O(1).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Perform `a[p] = operate(a[p], value)`.
    ///
    /// Time complexity is O(logn).
    pub fn update(&mut self, p: usize, value: T::S) {
        let p = p + 1; // 0-index to 1-index
        let ps = successors(Some(p), |&p| Some(p + lsb(p))).take_while(|&p| p <= self.n);
        for p in ps {
            self.node[p] = T::operate(&self.node[p], &value);
        }
    }

    /// Returns `operate(a[0], a[1], ..., a[p-1])`.
    ///
    /// Time complexity is O(logn).
    pub fn prefix(&self, p: usize) -> T::S {
        let ps = successors(Some(p), |&p| Some(p - lsb(p))).take_while(|&p| p > 0);
        let mut res = T::identity();
        for p in ps {
            res = T::operate(&res, &self.node[p]);
        }
        res
    }

    /// Find max `p` which satisfy `f(a[0], a[1], ..., a[p - 1])` or `p = 0` if no such `p` exist.
    ///
    /// `f(T::identity()) == true` must be held.
    ///
    /// Time complexity is O(log^2N).
    pub fn max_right<F: FnMut(T::S) -> bool>(&self, mut f: F) -> usize {
        // TODO: O(log^2N) => O(logN)
        let mut ok = 0;
        let mut ng = self.n + 1;
        while ng - ok > 1 {
            let mid = (ok + ng) / 2;
            if f(self.prefix(mid)) {
                ok = mid;
            } else {
                ng = mid;
            }
        }
        ok
    }
}

#[cfg(test)]
mod test {
    use super::{Additive, FenwickTree, Max, Min, Multiplicative};

    #[test]
    fn test_additive() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::<Additive<usize>>::from(a.clone());
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
        let mut ft = FenwickTree::<Multiplicative<usize>>::from(a.clone());
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
        let mut ft = FenwickTree::<Max<usize>>::from(a.clone());
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
        let mut ft = FenwickTree::<Min<usize>>::from(a.clone());
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

    #[test]
    fn test_from_iter_for_size_unknown_iterator() {
        let iter = std::iter::successors(Some(0), |&v| if v < 5 { Some(v + 1) } else { None });
        let ft = FenwickTree::<Additive<usize>>::from_iter(iter.clone());
        assert_eq!(ft.len(), iter.count());
    }

    #[test]
    fn test_max_right() {
        let ft = FenwickTree::<Additive<usize>>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(ft.max_right(|v| v <= 6), 3);
        assert_eq!(ft.max_right(|v| v <= 0), 0);
        assert_eq!(ft.max_right(|v| v <= 100), 5);
    }
}
