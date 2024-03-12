use std::{
    cmp::{max, min},
    marker::PhantomData,
    mem::replace,
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
    ($($type:ident),* $(,)*) => {$(
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

impl_int! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
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
        max(a.clone(), b.clone())
    }
}

pub struct Min<T>(PhantomData<fn() -> T>);

impl<T: MinIdentity + Clone> Monoid for Min<T> {
    type S = T;

    fn identity() -> Self::S {
        T::identity()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        min(a.clone(), b.clone())
    }
}

pub struct Segtree<T: Monoid> {
    node: Vec<T::S>,
    n: usize,
}

impl<T: Monoid> From<Vec<T::S>> for Segtree<T> {
    fn from(mut value: Vec<T::S>) -> Self {
        let mut st = Self::new(value.len());
        let base = st.leaf_base();
        for i in 0..value.len() {
            st.node[i + base] = replace(&mut value[i], T::identity());
        }
        for i in (0..base).rev() {
            st.node[i] = T::operate(&st.node[2 * i + 1], &st.node[2 * i + 2]);
        }
        st
    }
}

impl<T: Monoid> From<&[T::S]> for Segtree<T> {
    fn from(value: &[T::S]) -> Self {
        Self::from(value.to_vec())
    }
}

impl<T: Monoid> From<&Vec<T::S>> for Segtree<T> {
    fn from(value: &Vec<T::S>) -> Self {
        Self::from(value.clone())
    }
}

impl<T: Monoid> Segtree<T> {
    pub fn new(n: usize) -> Self {
        let len = 2 * n.next_power_of_two() - 1;
        let node = vec![T::identity(); len];
        Self { node, n }
    }

    pub fn set(&mut self, p: usize, value: T::S) {
        let base = self.leaf_base();
        self.node[p + base] = value;
        let mut i = p + base;
        while i > 0 {
            i = (i - 1) / 2;
            self.node[i] = T::operate(&self.node[2 * i + 1], &self.node[2 * i + 2]);
        }
    }

    pub fn get(&self, p: usize) -> &T::S {
        &self.node[p + self.leaf_base()]
    }

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
            if l % 2 == 0 {
                lv = T::operate(&lv, &self.node[l]);
                l += 1;
            }
            if r % 2 == 0 {
                r -= 1;
                rv = T::operate(&self.node[r], &rv);
            }
            l = (l - 1) / 2;
            r = (r - 1) / 2;
        }
        T::operate(&lv, &rv)
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
        let mut st = Segtree::<Additive<usize>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(st.query(..), 15);
        assert_eq!(st.query(1..), 15);
        st.set(0, 100);
        assert_eq!(st.query(..), 115);
        assert_eq!(st.query(1..), 15);
    }

    #[test]
    fn multiplicative() {
        let mut st = Segtree::<Multiplicative<usize>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(st.query(..), 0);
        assert_eq!(st.query(1..), 120);
        st.set(0, 2);
        assert_eq!(st.query(..), 240);
        assert_eq!(st.query(1..), 120);
    }

    #[test]
    fn max() {
        let mut st = Segtree::<Max<usize>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(st.query(..), 5);
        assert_eq!(st.query(..5), 4);
        st.set(0, 100);
        assert_eq!(st.query(..), 100);
        assert_eq!(st.query(1..5), 4);
    }

    #[test]
    fn min() {
        let mut st = Segtree::<Min<usize>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(st.query(..), 0);
        assert_eq!(st.query(1..), 1);
        st.set(0, 100);
        assert_eq!(st.query(..), 1);
        assert_eq!(st.query(2..), 2);
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
}
