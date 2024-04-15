use std::{
    cmp::{max_by, min_by, Ordering},
    fmt::Debug,
    ops::Add,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Compress<T> {
    map: Vec<T>,
}

impl<T: Debug> Debug for Compress<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.map.iter().zip(0..)).finish()
    }
}

impl<T: Ord> Compress<T> {
    /// Returns compressed given value if exist.
    ///
    /// Time complexity is O(logN).
    pub fn compress(&self, value: &T) -> Option<usize> {
        self.map.binary_search(value).ok()
    }

    /// Returns decompressed given value if exist.
    ///
    /// Time complexity is O(1).
    pub fn decompress(&self, value: usize) -> Option<&T> {
        self.map.get(value)
    }
}

/// Perform coordinate compression.
///
/// Time complexity is O(NlogN).
pub fn compress<T: Ord, I: IntoIterator<Item = T>>(iter: I) -> Compress<T> {
    let mut map = iter.into_iter().collect::<Vec<_>>();
    map.sort();
    map.dedup();
    Compress { map }
}

/// A struct which provide functionals to compress/decompress data with run-length encoding.
pub struct RLE;

impl RLE {
    /// Compress given data with run-length encoding.
    ///
    /// Time complexity is O(N).
    pub fn compress<T, I>(iter: I) -> Vec<(T, usize)>
    where
        T: Eq,
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut res = Vec::new();
        let mut prev = match iter.next() {
            Some(first) => first,
            None => return res,
        };
        let mut count = 1;
        for v in iter {
            if v != prev {
                res.push((prev, count));
                count = 0;
            }
            count += 1;
            prev = v;
        }
        res.push((prev, count));
        res
    }

    /// Decompress given run-length encoded data.
    ///
    /// Time complexity is O(N).
    pub fn decompress<T, I>(iter: I) -> Vec<T>
    where
        T: Clone,
        I: IntoIterator<Item = (T, usize)>,
    {
        let mut res = Vec::new();
        for (t, c) in iter {
            res.append(&mut vec![t; c]);
        }
        res
    }
}

/// Calculate maximum sum of subsequence of given iterator with respect to the specified
/// comparition function.
///
/// Time complexity is O(N).
pub fn max_subsequence_by<T, I, F>(iter: I, mut compare: F) -> Option<T>
where
    T: Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
    F: FnMut(&T, &T) -> Ordering,
{
    let mut iter = iter.into_iter();
    let mut dp = iter.next()?;
    let mut res = dp.clone();
    for value in iter {
        dp = max_by(dp + value.clone(), value, &mut compare);
        res = max_by(res, dp.clone(), &mut compare);
    }
    Some(res)
}

/// Calculate maximum sum of subsequence of given iterator with respect to the specified
/// key extraction function.
///
/// Time complexity is O(N).
pub fn max_subsequence_by_key<T, I, F, K>(iter: I, mut f: F) -> Option<T>
where
    T: Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
    F: FnMut(&T) -> K,
    K: Ord,
{
    max_subsequence_by(iter, |a, b| f(a).cmp(&f(b)))
}

/// Calculate maximum sum of subsequence of given iterator.
///
/// Time complexity is O(N).
pub fn max_subsequence<T, I>(iter: I) -> Option<T>
where
    T: Ord + Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
{
    max_subsequence_by(iter, |a, b| a.cmp(b))
}

/// Calculate minimum sum of subsequence of given iterator with respect to the specified
/// comparition function.
///
/// Time complexity is O(N).
pub fn min_subsequence_by<T, I, F>(iter: I, mut compare: F) -> Option<T>
where
    T: Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
    F: FnMut(&T, &T) -> Ordering,
{
    let mut iter = iter.into_iter();
    let mut dp = iter.next()?;
    let mut res = dp.clone();
    for value in iter {
        dp = min_by(dp + value.clone(), value, &mut compare);
        res = min_by(res, dp.clone(), &mut compare);
    }
    Some(res)
}

/// Calculate minimum sum of subsequence of given iterator with respect to the specified
/// key extraction function.
///
/// Time complexity is O(N).
pub fn min_subsequence_by_key<T, I, F, K>(iter: I, mut f: F) -> Option<T>
where
    T: Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
    F: FnMut(&T) -> K,
    K: Ord,
{
    min_subsequence_by(iter, |a, b| f(a).cmp(&f(b)))
}

/// Calculate minimum sum of subsequence of given iterator.
///
/// Time complexity is O(N).
pub fn min_subsequence<T, I>(iter: I) -> Option<T>
where
    T: Ord + Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
{
    min_subsequence_by(iter, |a, b| a.cmp(b))
}

/// Returns true if given `sub` is subsequence of `iter` with respect to the specified
/// comparition function.
///
/// The `sub` sequence need not be continueous at `iter`.
///
/// Time complexity is O(N).
pub fn is_subsequence_of_by<T, I1, I2, F>(iter: I1, sub: I2, mut f: F) -> bool
where
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = T>,
    F: FnMut(&T, &T) -> bool,
{
    let mut sub = sub.into_iter().peekable();
    for v in iter.into_iter() {
        if let Some(w) = sub.peek() {
            if f(&v, w) {
                sub.next();
            }
        }
    }
    sub.next().is_none()
}

/// Returns true if given `sub` is subsequence of `iter` with respect to the specified
/// key extraction function.
///
/// The `sub` sequence need not be continueous at `iter`.
///
/// Time complexity is O(N).
pub fn is_subsequence_of_by_key<T, I1, I2, F, K>(iter: I1, sub: I2, mut f: F) -> bool
where
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = T>,
    F: FnMut(&T) -> K,
    K: Eq,
{
    is_subsequence_of_by(iter, sub, |v1, v2| f(v1) == f(v2))
}

/// Returns true if given `sub` is subsequence of `iter`.
///
/// The `sub` sequence need not be continueous at `iter`.
///
/// Time complexity is O(N).
pub fn is_subsequence_of<T, I1, I2>(iter: I1, sub: I2) -> bool
where
    T: Eq,
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = T>,
{
    is_subsequence_of_by(iter, sub, |v1, v2| v1 == v2)
}

#[cfg(test)]
mod test {
    use super::{compress, is_subsequence_of, max_subsequence, min_subsequence, RLE};

    #[test]
    fn test_compress() {
        let v = vec![2, 100, 5, 3, 201, 4, 100];
        let c = compress(v.clone());
        assert_eq!(
            v.iter().map(|v| c.compress(v).unwrap()).collect::<Vec<_>>(),
            vec![0, 4, 3, 1, 5, 2, 4]
        );
        assert_eq!(
            vec![0, 4, 3, 1, 5, 2, 4]
                .into_iter()
                .map(|v| c.decompress(v).copied().unwrap())
                .collect::<Vec<_>>(),
            v
        );
    }

    #[test]
    fn test_rle_compress() {
        let v = vec![2, 100, 4, 4, 3, 2, 2, 2];
        assert_eq!(
            RLE::compress(v),
            vec![(2, 1), (100, 1), (4, 2), (3, 1), (2, 3)]
        );
        assert_eq!(RLE::compress(Vec::<usize>::new()), vec![]);
    }

    #[test]
    fn test_rle_decompress() {
        let v = vec![(2, 1), (100, 1), (4, 2), (3, 1), (2, 3)];
        assert_eq!(RLE::decompress(v), vec![2, 100, 4, 4, 3, 2, 2, 2]);
        assert_eq!(RLE::decompress(Vec::<(usize, usize)>::new()), vec![]);
    }

    #[test]
    fn test_rle_compress_decompress() {
        let v = vec![2, 100, 4, 4, 3, 2, 2, 2];
        assert_eq!(RLE::decompress(RLE::compress(v.clone())), v);
        assert_eq!(
            RLE::decompress(RLE::compress(Vec::<usize>::new())),
            Vec::new()
        );
    }

    #[test]
    fn test_max_subsequence() {
        let v = vec![1, -1, 3, -2, 4, -5];
        let mut max = isize::MIN;
        for i in 0..v.len() {
            for j in i..=v.len() {
                let sum = v[i..j].iter().sum();
                max = max.max(sum);
            }
        }
        assert_eq!(max_subsequence(v), Some(max));
        assert_eq!(max_subsequence(Vec::<isize>::new()), None);
    }

    #[test]
    fn test_min_subsequence() {
        let v = vec![1, -1, 3, -2, 4, -5];
        let mut min = isize::MAX;
        for i in 0..v.len() {
            for j in i..=v.len() {
                let sum = v[i..j].iter().sum();
                min = min.min(sum);
            }
        }
        assert_eq!(min_subsequence(v), Some(min));
        assert_eq!(min_subsequence(Vec::<isize>::new()), None);
    }

    #[test]
    fn test_is_subsequence_of() {
        assert!(is_subsequence_of(vec![1, 2, 3, 4], vec![2, 4]));
        assert!(is_subsequence_of(vec![1, 3, 4, 6, 8], vec![1, 4, 6, 8]));
        assert!(!is_subsequence_of(vec![1, 3, 4, 6, 8], vec![1, 1, 4, 6, 8]));
    }
}
