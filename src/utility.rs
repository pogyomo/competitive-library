use std::{collections::BTreeMap, hash::Hash};

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

#[cfg(test)]
mod test {
    use super::compress;

    #[test]
    fn test_compress() {
        let v = vec![2, 100, 5, 3, 201, 4, 100];
        assert_eq!(compress(v), vec![0, 4, 3, 1, 5, 2, 4]);
    }
}
