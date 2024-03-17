use std::{
    cmp::{max_by, min_by, Ordering},
    collections::BTreeMap,
    ops::Add,
};

/// Perform coordinate compression. Time complexity is O(NlogN).
pub fn compress<T: Ord + Clone>(v: Vec<T>) -> Vec<usize> {
    let mut w = v.clone();
    w.sort();
    w.dedup();
    let mut value_to_priority = BTreeMap::new();
    for (w, i) in w.into_iter().zip(0..) {
        value_to_priority.insert(w, i);
    }
    v.into_iter()
        .map(|value| value_to_priority.get(&value).copied().unwrap())
        .collect()
}

/// Compress given data with run-length encoding. Time complexity is O(N).
pub fn rle<T: Eq, I: IntoIterator<Item = T>>(iter: I) -> Vec<(T, usize)> {
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

/// Calculate maximum sum of subsequence of given iterator with respect to the specified
/// comparition function.
/// Time complexity is O(N).
pub fn max_subsequence_by<T, I, F>(iter: I, mut compare: F) -> Option<T>
where
    T: Ord + Clone + Add<T, Output = T>,
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

/// Calculate maximum sum of subsequence of given iterator.
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
/// Time complexity is O(N).
pub fn min_subsequence_by<T, I, F>(iter: I, mut compare: F) -> Option<T>
where
    T: Ord + Clone + Add<T, Output = T>,
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

/// Calculate minimum sum of subsequence of given iterator.
/// Time complexity is O(N).
pub fn min_subsequence<T, I>(iter: I) -> Option<T>
where
    T: Ord + Clone + Add<T, Output = T>,
    I: IntoIterator<Item = T>,
{
    min_subsequence_by(iter, |a, b| a.cmp(b))
}

#[cfg(test)]
mod test {
    use super::{compress, max_subsequence, min_subsequence, rle};

    #[test]
    fn test_compress() {
        let v = vec![2, 100, 5, 3, 201, 4, 100];
        assert_eq!(compress(v), vec![0, 4, 3, 1, 5, 2, 4]);
    }

    #[test]
    fn test_rle() {
        let v = vec![2, 100, 4, 4, 3, 2, 2, 2];
        assert_eq!(rle(v), vec![(2, 1), (100, 1), (4, 2), (3, 1), (2, 3)]);
        assert_eq!(rle(Vec::<usize>::new()), vec![]);
    }

    #[test]
    fn test_max_subsequence() {
        let v = vec![1, -1, 3, -2, 4, -5];
        let mut max = isize::MIN;
        for i in 0..v.len() {
            for j in i..v.len() {
                let mut sum = 0;
                for k in i..=j {
                    sum += v[k];
                }
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
            for j in i..v.len() {
                let mut sum = 0;
                for k in i..=j {
                    sum += v[k];
                }
                min = min.min(sum);
            }
        }
        assert_eq!(min_subsequence(v), Some(min));
        assert_eq!(min_subsequence(Vec::<isize>::new()), None);
    }
}
