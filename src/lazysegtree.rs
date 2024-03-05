pub use super::segtree::{Additive, Max, Min, Monoid, Multiplicative};
use std::ops::{Bound, RangeBounds};

/// Convert given range into [l, r).
fn range_normalize<R: RangeBounds<usize>>(range: R, min: usize, max: usize) -> (usize, usize) {
    let l = match range.start_bound() {
        Bound::Unbounded => min,
        Bound::Included(&l) => l,
        Bound::Excluded(&l) => l + 1,
    };
    let r = match range.end_bound() {
        Bound::Unbounded => max,
        Bound::Included(&r) => r + 1, // convert ..=r to ..r+1
        Bound::Excluded(&r) => r,
    };
    (l, r)
}

pub trait MapMonoid {
    type S: Monoid;
    type F: Clone + Eq;
    fn map_identity() -> Self::F;
    fn apply(f: &Self::F, x: &<Self::S as Monoid>::S) -> <Self::S as Monoid>::S;

    /// Create h(x) = g(f(x))
    fn composition(f: &Self::F, g: &Self::F) -> Self::F;
}

pub struct LazySegtree<T: MapMonoid> {
    node: Vec<<T::S as Monoid>::S>,
    lazy: Vec<T::F>,
    n: usize,
}

impl<T: MapMonoid> From<&[<T::S as Monoid>::S]> for LazySegtree<T> {
    fn from(value: &[<T::S as Monoid>::S]) -> Self {
        let mut st = Self::new(value.len());
        let base = st.leaf_base();
        for i in 0..value.len() {
            st.node[i + base] = value[i].clone();
        }
        for i in (0..base).rev() {
            st.node[i] = T::S::operate(&st.node[2 * i + 1], &st.node[2 * i + 2]);
        }
        st
    }
}

impl<T: MapMonoid> From<&Vec<<T::S as Monoid>::S>> for LazySegtree<T> {
    fn from(value: &Vec<<T::S as Monoid>::S>) -> Self {
        Self::from(value.as_slice())
    }
}

impl<T: MapMonoid> From<Vec<<T::S as Monoid>::S>> for LazySegtree<T> {
    fn from(value: Vec<<T::S as Monoid>::S>) -> Self {
        Self::from(value.as_slice())
    }
}

impl<T: MapMonoid> LazySegtree<T> {
    pub fn new(n: usize) -> Self {
        let len = 2 * n.next_power_of_two() - 1;
        let node = vec![T::S::identity(); len];
        let lazy = vec![T::map_identity(); len];
        Self { node, lazy, n }
    }

    pub fn get(&mut self, p: usize) -> &<T::S as Monoid>::S {
        assert!(p < self.n);
        let p = p + self.leaf_base();
        self.propagate_recursive(p);
        &self.node[p]
    }

    pub fn set(&mut self, p: usize, value: <T::S as Monoid>::S) {
        let mut p = p + self.leaf_base();
        self.propagate_recursive(p);
        self.node[p] = value;
        while p > 0 {
            p = (p - 1) / 2;
            self.node[p] = T::S::operate(&self.node[2 * p + 1], &self.node[2 * p + 2]);
        }
    }

    pub fn apply<R: RangeBounds<usize>>(&mut self, range: R, f: T::F) {
        enum StackState {
            Update(usize, usize, usize),
            Merge(usize),
        }

        let (l, r) = range_normalize(range, 0, self.n);
        let mut stack = Vec::new();
        stack.push(StackState::Update(0, 0, self.leaf_len()));
        while let Some(state) = stack.pop() {
            match state {
                StackState::Update(p, pl, pr) => {
                    self.propagate(p);
                    if pr <= l || r <= pl {
                        continue;
                    }
                    if l <= pl && pr <= r {
                        self.lazy[p] = f.clone();
                        self.propagate(p);
                    } else {
                        let mid = (pl + pr) / 2;
                        stack.push(StackState::Merge(p));
                        stack.push(StackState::Update(p * 2 + 2, mid, pr));
                        stack.push(StackState::Update(p * 2 + 1, pl, mid));
                    }
                }
                StackState::Merge(p) => {
                    self.node[p] = T::S::operate(&self.node[p * 2 + 1], &self.node[p * 2 + 2]);
                }
            }
        }
    }

    pub fn query<R: RangeBounds<usize>>(&mut self, range: R) -> <T::S as Monoid>::S {
        enum StackState {
            Query(usize, usize, usize),
            Value(usize),
        }

        let (l, r) = range_normalize(range, 0, self.n);
        let mut res = T::S::identity();
        let mut stack = Vec::new();
        stack.push(StackState::Query(0, 0, self.leaf_len()));
        while let Some(state) = stack.pop() {
            match state {
                StackState::Query(p, pl, pr) => {
                    self.propagate(p);
                    if pr <= l || r <= pl {
                        continue;
                    }
                    if l <= pl && pr <= r {
                        stack.push(StackState::Value(p));
                    } else {
                        let mid = (pl + pr) / 2;
                        stack.push(StackState::Query(p * 2 + 2, mid, pr));
                        stack.push(StackState::Query(p * 2 + 1, pl, mid));
                    }
                }
                StackState::Value(p) => {
                    res = T::S::operate(&res, &self.node[p]);
                }
            }
        }
        res
    }

    /// Apply lazy[p] to node[p] and propagate the lazy value to childs if exist.
    fn propagate(&mut self, p: usize) {
        if self.lazy[p] == T::map_identity() {
            return;
        }
        if p < self.leaf_base() {
            self.lazy[p * 2 + 1] = T::composition(&self.lazy[p], &self.lazy[p * 2 + 1]);
            self.lazy[p * 2 + 2] = T::composition(&self.lazy[p], &self.lazy[p * 2 + 2]);
        }
        self.node[p] = T::apply(&self.lazy[p], &self.node[p]);
        self.lazy[p] = T::map_identity();
    }

    /// Call propagate recursively:
    /// propagate(0 = p_0) => propagate(p_1) => propagate(p_2) => ... => propagate(p_n = p)
    /// where p_i = (p_{i+1} - 1) / 2 for 0 <= i < n
    fn propagate_recursive(&mut self, mut p: usize) {
        let mut visit = Vec::new();
        visit.push(p);
        while p > 0 {
            p = (p - 1) / 2;
            visit.push(p);
        }
        for p in visit {
            self.propagate(p);
        }
    }

    fn leaf_len(&self) -> usize {
        self.n.next_power_of_two()
    }

    fn leaf_base(&self) -> usize {
        self.n.next_power_of_two() - 1
    }
}

#[cfg(test)]
mod test {
    use super::{LazySegtree, MapMonoid, Max, Monoid};
    use std::marker::PhantomData;

    struct MapAdditive<S>(PhantomData<fn() -> S>);

    impl<S: Monoid<S = usize>> MapMonoid for MapAdditive<S> {
        type S = S;
        type F = usize;
        fn map_identity() -> Self::F {
            0
        }
        fn apply(f: &Self::F, x: &<Self::S as super::Monoid>::S) -> <Self::S as super::Monoid>::S {
            f + x
        }
        fn composition(f: &Self::F, g: &Self::F) -> Self::F {
            f + g
        }
    }

    #[test]
    fn max_and_additive_with_apply_and_query() {
        let mut st = LazySegtree::<MapAdditive<Max<usize>>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(st.query(..), 5);
        assert_eq!(st.query(2..4), 3);
        st.apply(2..4, 3);
        assert_eq!(st.query(..), 6);
        assert_eq!(st.query(2..4), 6);
    }

    #[test]
    fn max_and_additive_with_apply_and_get() {
        let mut st = LazySegtree::<MapAdditive<Max<usize>>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        st.apply(2..4, 3);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 5, 6, 4, 5]
        );
    }

    #[test]
    fn max_and_additive_with_set_and_query() {
        let mut st = LazySegtree::<MapAdditive<Max<usize>>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(st.query(..), 5);
        assert_eq!(st.query(2..4), 3);
        st.set(2, 6);
        assert_eq!(st.query(..), 6);
        assert_eq!(st.query(2..4), 6);
    }

    #[test]
    fn max_and_additive_with_set_and_get() {
        let mut st = LazySegtree::<MapAdditive<Max<usize>>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        st.set(2, 6);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 6, 3, 4, 5]
        );
    }

    #[test]
    fn max_and_additive_with_all_operation() {
        let mut st = LazySegtree::<MapAdditive<Max<usize>>>::from(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(st.query(..), 5);
        assert_eq!(st.query(2..4), 3);
        st.apply(2..4, 3);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 5, 6, 4, 5]
        );
        assert_eq!(st.query(..), 6);
        assert_eq!(st.query(2..4), 6);
        st.set(2, 7);
        assert_eq!(
            (0..6).map(|i| *st.get(i)).collect::<Vec<_>>(),
            vec![0, 1, 7, 6, 4, 5]
        );
        assert_eq!(st.query(..), 7);
        assert_eq!(st.query(2..4), 7);
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

        struct EmptyMat2MapMonoid;

        impl MapMonoid for EmptyMat2MapMonoid {
            type S = Mat2Monoid;
            type F = ();
            fn apply(_: &Self::F, x: &<Self::S as Monoid>::S) -> <Self::S as Monoid>::S {
                *x
            }
            fn map_identity() -> Self::F {
                ()
            }
            fn composition(_: &Self::F, _: &Self::F) -> Self::F {
                ()
            }
        }

        let m1 = Mat2([[1, 2], [3, 4]]);
        let m2 = Mat2([[4, 3], [2, 1]]);
        let m3 = Mat2([[0, 5], [8, 2]]);
        let mut st = LazySegtree::<EmptyMat2MapMonoid>::new(3);
        assert_eq!(st.query(..), Mat2::identity());
        st.set(0, m1);
        st.set(1, m2);
        st.set(2, m3);
        assert_eq!(st.query(..), m1 * m2 * m3);
        assert_eq!(st.query(0..2), m1 * m2);
        assert_eq!(st.query(1..3), m2 * m3);
    }
}
