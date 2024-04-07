/// A struct to manage disjoint sets.
pub struct UnionFind {
    parent: Vec<Option<usize>>,
    size: Vec<usize>,
    n: usize,
}

impl UnionFind {
    /// Create a `UnionFind` with size `n`.
    pub fn new(n: usize) -> Self {
        let parent = vec![None; n];
        let size = vec![1; n];
        Self { parent, size, n }
    }

    /// Returns the root element of the group to which the element belongs.
    ///
    /// Amortized time complexity is O(α(n)).
    pub fn root(&mut self, x: usize) -> usize {
        assert!(x < self.n);
        match self.parent[x] {
            Some(parent) => {
                let res = self.root(parent);
                self.parent[x] = Some(res);
                res
            }
            None => x,
        }
    }

    /// Returns true if the two elements is in same group.
    ///
    /// Amortized time complexity is O(α(n)).
    pub fn is_same(&mut self, x: usize, y: usize) -> bool {
        assert!(x < self.n && y < self.n);
        self.root(x) == self.root(y)
    }

    /// Merge two element. If the two elemnts is already in same group, return false.
    /// Otherwise return true.
    ///
    /// Amortized time complexity is O(α(n)).
    pub fn merge(&mut self, x: usize, y: usize) -> bool {
        assert!(x < self.n && y < self.n);
        let rx = self.root(x);
        let ry = self.root(y);
        if rx == ry {
            return false;
        }
        // union by size
        if self.size[rx] < self.size[ry] {
            self.parent[rx] = Some(ry);
            self.size[ry] += self.size[rx];
        } else {
            self.parent[ry] = Some(rx);
            self.size[rx] += self.size[ry];
        }
        true
    }

    /// Returns the size of group to which the element belongs.
    ///
    /// Time complexity is O(1).
    pub fn size(&self, x: usize) -> usize {
        assert!(x < self.n);
        self.size[x]
    }
}

#[cfg(test)]
mod test {
    use super::UnionFind;

    #[test]
    fn test() {
        let mut uf = UnionFind::new(5);
        uf.merge(1, 2);
        uf.merge(3, 4);
        assert!(uf.is_same(1, 2));
        assert!(uf.is_same(3, 4));
        assert!(!uf.is_same(0, 1));
        uf.merge(0, 3);
        assert!(uf.is_same(0, 3));
        assert!(!uf.is_same(0, 1));
    }
}
