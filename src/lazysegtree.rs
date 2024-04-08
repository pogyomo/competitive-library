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

impl<T: MapMonoid> FromIterator<<T::S as Monoid>::S> for LazySegtree<T> {
    fn from_iter<I: IntoIterator<Item = <T::S as Monoid>::S>>(iter: I) -> Self {
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
        let lazy = vec![T::map_identity(); n2 + n2 - 1];
        let mut node = Vec::with_capacity(n2 + n2 - 1);
        node.resize(n2 - 1, T::S::identity());
        node.extend(iter);
        node.resize(n2 + n2 - 1, T::S::identity());
        for i in (0..n2 - 1).rev() {
            node[i] = T::S::operate(&node[(i << 1) + 1], &node[(i << 1) + 2]);
        }
        Self { node, lazy, n }
    }
}

impl<T: MapMonoid> From<Vec<<T::S as Monoid>::S>> for LazySegtree<T> {
    fn from(value: Vec<<T::S as Monoid>::S>) -> Self {
        Self::from_iter(value)
    }
}

impl<T: MapMonoid> LazySegtree<T> {
    /// Construct a new `LazySegtree` with length `n`.
    pub fn new(n: usize) -> Self {
        let len = (n.next_power_of_two() << 1) - 1;
        let node = vec![T::S::identity(); len];
        let lazy = vec![T::map_identity(); len];
        Self { node, lazy, n }
    }

    /// Returns the size of `LazySegtree`.
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

    /// Returns `a[p]`.
    ///
    /// Time complexity is O(logn).
    pub fn get(&mut self, p: usize) -> &<T::S as Monoid>::S {
        assert!(p < self.n);
        let p = p + self.leaf_base();
        self.propagate_recursive(p);
        &self.node[p]
    }

    /// Perform `a[p] = value`.
    ///
    /// Time complexity is O(logn).
    pub fn set(&mut self, p: usize, value: <T::S as Monoid>::S) {
        let mut p = p + self.leaf_base();
        self.propagate_recursive(p);
        self.node[p] = value;
        while p > 0 {
            p = (p - 1) >> 1;
            self.node[p] = T::S::operate(&self.node[(p << 1) + 1], &self.node[(p << 1) + 2]);
        }
    }

    /// Perform `a[range] <= f(a[range])`.
    ///
    /// Time complexity is O(logn).
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
                        let mid = (pl + pr) >> 1;
                        stack.push(StackState::Merge(p));
                        stack.push(StackState::Update((p << 1) + 2, mid, pr));
                        stack.push(StackState::Update((p << 1) + 1, pl, mid));
                    }
                }
                StackState::Merge(p) => {
                    self.node[p] =
                        T::S::operate(&self.node[(p << 1) + 1], &self.node[(p << 1) + 2]);
                }
            }
        }
    }

    /// Calculate `operate(a[range])`.
    ///
    /// Time complexity is O(logn).
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
                        let mid = (pl + pr) >> 1;
                        stack.push(StackState::Query((p << 1) + 2, mid, pr));
                        stack.push(StackState::Query((p << 1) + 1, pl, mid));
                    }
                }
                StackState::Value(p) => {
                    res = T::S::operate(&res, &self.node[p]);
                }
            }
        }
        res
    }

    /// Find max `r > l` which satisfy `f(a[l], a[l + 1], ..., a[r - 1])` or `r = l` if no such `r` exist.
    ///
    /// `f(T::S::identity()) == true` must be held.
    ///
    /// Time complexity is O(log^2n).
    pub fn max_right<F>(&mut self, l: usize, mut f: F) -> usize
    where
        F: FnMut(<T::S as Monoid>::S) -> bool,
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
    /// `f(T::S::identity()) == true` must be held.
    ///
    /// Time complexity is O(log^2n).
    pub fn min_left<F>(&mut self, r: usize, mut f: F) -> usize
    where
        F: FnMut(<T::S as Monoid>::S) -> bool,
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

    /// Apply lazy[p] to node[p] and propagate the lazy value to childs if exist.
    fn propagate(&mut self, p: usize) {
        if self.lazy[p] == T::map_identity() {
            return;
        }
        if p < self.leaf_base() {
            self.lazy[(p << 1) + 1] = T::composition(&self.lazy[p], &self.lazy[(p << 1) + 1]);
            self.lazy[(p << 1) + 2] = T::composition(&self.lazy[p], &self.lazy[(p << 1) + 2]);
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
            p = (p - 1) >> 1;
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
    use super::{Additive, LazySegtree, MapMonoid, Max, Monoid};
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

    struct EmptyMapMonoid<M>(PhantomData<fn() -> M>);

    impl<M: Monoid> MapMonoid for EmptyMapMonoid<M> {
        type S = M;
        type F = ();
        fn apply(_: &Self::F, x: &<Self::S as Monoid>::S) -> <Self::S as Monoid>::S {
            x.clone()
        }
        fn map_identity() -> Self::F {
            ()
        }
        fn composition(_: &Self::F, _: &Self::F) -> Self::F {
            ()
        }
    }

    #[test]
    fn max_map_additive() {
        let mut v = vec![0, 1, 2, 3, 4, 5];
        let mut st = LazySegtree::<MapAdditive<Max<usize>>>::from(v.clone());
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().copied().max().unwrap_or(0));
            }
        }
        st.apply(1..3, 4);
        for i in 1..3 {
            v[i] += 4;
        }
        for i in 0..=v.len() {
            for j in i..=v.len() {
                assert_eq!(st.query(i..j), v[i..j].iter().copied().max().unwrap_or(0));
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

    #[test]
    fn test_from_iter_for_size_unknown_iterator() {
        let iter = std::iter::successors(Some(0), |&v| if v < 5 { Some(v + 1) } else { None });
        let ft = LazySegtree::<MapAdditive<Max<usize>>>::from_iter(iter.clone());
        assert_eq!(ft.len(), iter.count());
    }

    #[test]
    fn test_max_right() {
        let mut st = LazySegtree::<EmptyMapMonoid<Additive<usize>>>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(st.max_right(0, |v| v <= 6), 3);
        assert_eq!(st.max_right(0, |v| v <= 0), 0);
        assert_eq!(st.max_right(0, |v| v <= 100), 5);
    }

    #[test]
    fn test_min_left() {
        let mut st = LazySegtree::<EmptyMapMonoid<Additive<usize>>>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(st.min_left(5, |v| v <= 12), 2);
        assert_eq!(st.min_left(5, |v| v <= 4), 5);
        assert_eq!(st.min_left(5, |v| v <= 100), 0);
    }
}
