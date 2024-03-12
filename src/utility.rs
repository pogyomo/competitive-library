use std::collections::BTreeMap;

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

#[cfg(test)]
mod test {
    use super::{compress, rle};

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
}
