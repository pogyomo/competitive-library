use std::{
    cmp::{max, min},
    marker::PhantomData,
    ops::{Add, Bound, Mul, RangeBounds},
};

/// A helper trait for type which has identity of additive.
trait AdditiveIdentity: Add<Self, Output = Self> + Sized {
    fn id() -> Self;
}

/// A helper trait for type which has identity of multiplicative.
trait MultiplicativeIdentity: Mul<Self, Output = Self> + Sized {
    fn id() -> Self;
}

/// A helper trait for type which has identity of max.
trait MaxIdentity: Ord {
    fn id() -> Self;
}

/// A helper trait for type which has identity of min.
trait MinIdentity: Ord {
    fn id() -> Self;
}

macro_rules! impl_int {
    ($($type:ident),* $(,)*) => {$(
        impl AdditiveIdentity for $type {
            fn id() -> Self { 0 }
        }

        impl MultiplicativeIdentity for $type {
            fn id() -> Self { 1 }
        }

        impl MaxIdentity for $type {
            fn id() -> Self { $type::MIN }
        }

        impl MinIdentity for $type {
            fn id() -> Self { $type::MAX }
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
        T::id()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        a.clone() + b.clone()
    }
}

pub struct Multiplicative<T>(PhantomData<fn() -> T>);

impl<T: MultiplicativeIdentity + Clone> Monoid for Multiplicative<T> {
    type S = T;

    fn identity() -> Self::S {
        T::id()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        a.clone() * b.clone()
    }
}

pub struct Max<T>(PhantomData<fn() -> T>);

impl<T: MaxIdentity + Clone> Monoid for Max<T> {
    type S = T;

    fn identity() -> Self::S {
        T::id()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        max(a.clone(), b.clone())
    }
}

pub struct Min<T>(PhantomData<fn() -> T>);

impl<T: MinIdentity + Clone> Monoid for Min<T> {
    type S = T;

    fn identity() -> Self::S {
        T::id()
    }

    fn operate(a: &Self::S, b: &Self::S) -> Self::S {
        min(a.clone(), b.clone())
    }
}

pub struct Segtree<T: Monoid> {
    node: Vec<T::S>,
    n: usize,
}

impl<T: Monoid> From<&[T::S]> for Segtree<T> {
    fn from(value: &[T::S]) -> Self {
        let mut st = Self::new(value.len());
        let base = st.leaf_base();
        for i in 0..value.len() {
            st.node[i + base] = value[i].clone();
        }
        for i in (0..base).rev() {
            st.node[i] = T::operate(&st.node[2 * i + 1], &st.node[2 * i + 2]);
        }
        st
    }
}

impl<T: Monoid> From<&Vec<T::S>> for Segtree<T> {
    fn from(value: &Vec<T::S>) -> Self {
        Self::from(value.as_slice())
    }
}

impl<T: Monoid> From<Vec<T::S>> for Segtree<T> {
    fn from(value: Vec<T::S>) -> Self {
        Self::from(value.as_slice())
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
        let mut res = T::identity();
        while l < r {
            if l % 2 == 0 {
                res = T::operate(&self.node[l], &res);
                l += 1;
            }
            if r % 2 == 0 {
                r -= 1;
                res = T::operate(&res, &self.node[r]);
            }
            l = (l - 1) / 2;
            r = (r - 1) / 2;
        }
        res
    }

    fn leaf_base(&self) -> usize {
        self.n.next_power_of_two() - 1
    }
}

#[cfg(test)]
mod test {
    use super::{Additive, Max, Min, Multiplicative, Segtree};

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
}
