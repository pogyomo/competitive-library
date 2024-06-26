use std::{
    cmp::{max, min},
    marker::PhantomData,
    ops::{Add, Bound, Mul, RangeBounds},
};

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

pub trait Monoid {
    type S: Clone;
    fn identity() -> Self::S;
    fn operate(a: &Self::S, b: &Self::S) -> Self::S;
}

pub struct Additive<T>(PhantomData<fn() -> T>);

impl<T: AdditiveIdentity + Clone> Monoid for Additive<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        a.clone() + b.clone()
    }
}

pub struct Multiplicative<T>(PhantomData<fn() -> T>);

impl<T: MultiplicativeIdentity + Clone> Monoid for Multiplicative<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        a.clone() * b.clone()
    }
}

pub struct Max<T>(PhantomData<fn() -> T>);

impl<T: MaxIdentity + Clone> Monoid for Max<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        max(a, b).clone()
    }
}

pub struct Min<T>(PhantomData<fn() -> T>);

impl<T: MinIdentity + Clone> Monoid for Min<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        min(a, b).clone()
    }
}

pub struct Segtree<T: Monoid> {
    node: Vec<T::S>,
    n: usize,
}

impl<T: Monoid> FromIterator<T::S> for Segtree<T> {
    fn from_iter<I: IntoIterator<Item = T::S>>(iter: I) -> Self {
        enum EitherIter<T, I1: Iterator<Item = T>, I2: Iterator<Item = T>> {
            Left(I1),
            Right(I2),
        }

        impl<T, I1: Iterator<Item = T>, I2: Iterator<Item = T>> Iterator for EitherIter<T, I1, I2> {
            type Item = T;
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::Left(iter) => iter.next(),
                    Self::Right(iter) => iter.next(),
                }
            }
        }

        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let (n, n2, iter) = if upper == Some(lower) {
            let n = lower;
            let n2 = n.next_power_of_two();
            let iter = EitherIter::Left(iter);
            (n, n2, iter)
        } else {
            let v = iter.collect::<Vec<_>>();
            let n = v.len();
            let n2 = n.next_power_of_two();
            let iter = EitherIter::Right(v.into_iter());
            (n, n2, iter)
        };
        let mut node = Vec::with_capacity(n2 + n2 - 1);
        node.resize(n2 - 1, T::identity());
        node.extend(iter);
        node.resize(n2 + n2 - 1, T::identity());
        for i in (0..n2 - 1).rev() {
            node[i] = T::operate(&node[(i << 1) + 1], &node[(i << 1) + 2]);
        }
        Self { node, n }
    }
}

impl<T: Monoid> From<Vec<T::S>> for Segtree<T> {
    fn from(value: Vec<T::S>) -> Self {
        Self::from_iter(value)
    }
}

impl<T: Monoid> Segtree<T> {
    /// Construct a new `Segtree` with length `n`.
    pub fn new(n: usize) -> Self {
        let len = (n.next_power_of_two() << 1) - 1;
        let node = vec![T::identity(); len];
        Self { node, n }
    }

    /// Returns the size of `Segtree`.
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

    /// Perform `a[p] = value`.
    ///
    /// Time complexity is O(logn).
    pub fn set(&mut self, p: usize, value: T::S) {
        let base = self.leaf_base();
        self.node[p + base] = value;
        let mut i = p + base;
        while i > 0 {
            i = (i - 1) >> 1;
            self.node[i] = T::operate(&self.node[(i << 1) + 1], &self.node[(i << 1) + 2]);
        }
    }

    /// Returns `a[p]`.
    ///
    /// Time complexity is O(1).
    pub fn get(&self, p: usize) -> &T::S {
        &self.node[p + self.leaf_base()]
    }

    /// Returns `operate(a[range])`.
    ///
    /// Time complexity is O(logn).
    pub fn query<R: RangeBounds<usize>>(&self, range: R) -> T::S {
        let mut l = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&l) => l,
            Bound::Excluded(&l) => l + 1,
        } + self.leaf_base();
        let mut r = match range.end_bound() {
            Bound::Unbounded => self.n,
            Bound::Included(&r) => r + 1, // convert ..=r to ..r+1
            Bound::Excluded(&r) => r,
        } + self.leaf_base();
        let mut lv = T::identity();
        let mut rv = T::identity();
        while l < r {
            if l & 1 == 0 {
                lv = T::operate(&lv, &self.node[l]);
                l += 1;
            }
            if r & 1 == 0 {
                r -= 1;
                rv = T::operate(&self.node[r], &rv);
            }
            l = (l - 1) >> 1;
            r = (r - 1) >> 1;
        }
        T::operate(&lv, &rv)
    }

    /// Find max `r > l` which satisfy `f(a[l], a[l + 1], ..., a[r - 1])` or `r = l` if no such `r` exist.
    ///
    /// `f(T::identity()) == true` must be held.
    ///
    /// Time complexity is O(log^2N).
    pub fn max_right<F>(&self, l: usize, mut f: F) -> usize
    where
        F: FnMut(T::S) -> bool,
    {
        // TODO: O(log^2N) => O(logN)
        let mut ng = self.n + 1;
        let mut ok = l;
        while ng - ok > 1 {
            let mid = (ok + ng) / 2;
            if f(self.query(l..mid)) {
                ok = mid;
            } else {
                ng = mid;
            }
        }
        ok
    }

    /// Find min `l < r` which satisfy `f(a[l], a[l + 1], ..., a[r - 1])` or `l = r` if no such `l` exist.
    ///
    /// `f(T::identity()) == true` must be held.
    ///
    /// Time complexity is O(log^2N).
    pub fn min_left<F>(&self, r: usize, mut f: F) -> usize
    where
        F: FnMut(T::S) -> bool,
    {
        // TODO: O(log^2N) => O(logN)
        if f(self.query(0..r)) {
            return 0;
        }
        let mut ng = 0;
        let mut ok = r;
        while ok - ng > 1 {
            let mid = (ok + ng) / 2;
            if f(self.query(mid..r)) {
                ok = mid;
            } else {
                ng = mid;
            }
        }
        ok
    }

    fn leaf_base(&self) -> usize {
        self.n.next_power_of_two() - 1
    }
}

#[cfg(test)]
mod test {
    use super::{Additive, Max, Min, Monoid, Multiplicative, Segtree};

    #[test]
    fn additive() {
        let mut v = vec![0, 1, 2, 3, 4, 5];
        let mut st = Segtree::<Additive<usize>>::from(v.clone());
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().sum());
            }
        }
        st.set(0, 100);
        v[0] = 100;
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().sum());
            }
        }
    }

    #[test]
    fn multiplicative() {
        let mut v = vec![0, 1, 2, 3, 4, 5];
        let mut st = Segtree::<Multiplicative<usize>>::from(v.clone());
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().product());
            }
        }
        st.set(0, 100);
        v[0] = 100;
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().product());
            }
        }
    }

    #[test]
    fn max() {
        let mut v = vec![0, 1, 2, 3, 4, 5];
        let mut st = Segtree::<Max<usize>>::from(v.clone());
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().max().copied().unwrap_or(0));
            }
        }
        st.set(0, 100);
        v[0] = 100;
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().max().copied().unwrap_or(0));
            }
        }
    }

    #[test]
    fn min() {
        let mut v = vec![0, 1, 2, 3, 4, 5];
        let mut st = Segtree::<Min<usize>>::from(v.clone());
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(
                    st.query(i..j),
                    v[i..j].iter().min().copied().unwrap_or(usize::MAX)
                );
            }
        }
        st.set(0, 100);
        v[0] = 100;
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(
                    st.query(i..j),
                    v[i..j].iter().min().copied().unwrap_or(usize::MAX)
                );
            }
        }
    }

    #[test]
    fn check_commutative_law_is_not_required() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        struct Mat2([[usize; 2]; 2]);

        impl Mat2 {
            fn identity() -> Self {
                Mat2([[1, 0], [0, 1]])
            }
        }

        impl std::ops::Mul for Mat2 {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self::Output {
                let mut mat = [[0; 2]; 2];
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..2 {
                            mat[i][j] += self.0[i][k] * rhs.0[k][j];
                        }
                    }
                }
                Self(mat)
            }
        }

        struct Mat2Monoid;

        impl Monoid for Mat2Monoid {
            type S = Mat2;
            fn identity() -> Self::S {
                Mat2::identity()
            }
            fn operate(a: &Self::S, b: &Self::S) -> Self::S {
                *a * *b
            }
        }

        let m1 = Mat2([[1, 2], [3, 4]]);
        let m2 = Mat2([[4, 3], [2, 1]]);
        let m3 = Mat2([[0, 5], [8, 2]]);
        let mut st = Segtree::<Mat2Monoid>::new(3);
        assert_eq!(st.query(..), Mat2::identity());
        st.set(0, m1);
        st.set(1, m2);
        st.set(2, m3);
        assert_eq!(st.query(..), m1 * m2 * m3);
        assert_eq!(st.query(0..2), m1 * m2);
        assert_eq!(st.query(1..3), m2 * m3);
    }

    #[test]
    fn test_from_iter_for_size_unknown_iterator() {
        let iter = std::iter::successors(Some(0), |&v| if v < 5 { Some(v + 1) } else { None });
        let ft = Segtree::<Additive<usize>>::from_iter(iter.clone());
        assert_eq!(ft.len(), iter.count());
    }

    #[test]
    fn test_max_right() {
        let st = Segtree::<Additive<usize>>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(st.max_right(0, |v| v <= 6), 3);
        assert_eq!(st.max_right(0, |v| v <= 0), 0);
        assert_eq!(st.max_right(0, |v| v <= 100), 5);
    }

    #[test]
    fn test_min_left() {
        let st = Segtree::<Additive<usize>>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(st.min_left(5, |v| v <= 12), 2);
        assert_eq!(st.min_left(5, |v| v <= 4), 5);
        assert_eq!(st.min_left(5, |v| v <= 100), 0);
    }
}
