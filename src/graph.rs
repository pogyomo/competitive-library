use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, VecDeque},
    hash::Hash,
    ops::Add,
};

/// A trait represent directed/undirected graph with/without weight.
pub trait Graph {
    type V;
    type W;
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)>;
}

impl Graph for Vec<Vec<usize>> {
    type V = usize;
    type W = ();
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self[v].iter().copied().map(|v| (v, ())).collect()
    }
}

impl<W: Clone> Graph for Vec<Vec<(usize, W)>> {
    type V = usize;
    type W = W;
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self[v].clone()
    }
}

impl<V: Clone + Hash + Eq> Graph for HashMap<V, Vec<V>> {
    type V = V;
    type W = ();
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self.get(&v)
            .map(|vs| vs.into_iter().cloned().map(|v| (v, ())).collect())
            .unwrap_or(Vec::new())
    }
}

impl<V: Clone + Hash + Eq, W: Clone> Graph for HashMap<V, Vec<(V, W)>> {
    type V = V;
    type W = W;
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self.get(&v).map(|vs| vs.clone()).unwrap_or(Vec::new())
    }
}

impl<V: Clone + Ord> Graph for BTreeMap<V, Vec<V>> {
    type V = V;
    type W = ();
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self.get(&v)
            .map(|vs| vs.into_iter().cloned().map(|v| (v, ())).collect())
            .unwrap_or(Vec::new())
    }
}

impl<V: Clone + Ord, W: Clone> Graph for BTreeMap<V, Vec<(V, W)>> {
    type V = V;
    type W = W;
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self.get(&v).map(|vs| vs.clone()).unwrap_or(Vec::new())
    }
}

/// A trait to calculate single source shortest path distance of a graph by considering all vertex
/// distance is 1.
pub trait BFS: Graph
where
    Self::V: Clone,
{
    /// Calculate minimum number of edge in path from `start` to all vertex.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O(V + E).
    fn bfs<D>(&self, start: Self::V, mut dist: D) -> D
    where
        D: DistanceTable<Self::V, usize>,
    {
        let mut queue = VecDeque::new();
        queue.push_back(start.clone());
        dist.set_distance(start, 0);
        while let Some(u) = queue.pop_front() {
            for (v, _) in self.childs(u.clone()) {
                if dist.distance(&v).is_none() {
                    dist.set_distance(v.clone(), dist.distance(&u).unwrap() + 1);
                    queue.push_back(v);
                }
            }
        }
        dist
    }
}

impl<G: Graph> BFS for G where G::V: Clone {}

/// A trait to calculate single source shortest path distance of a graph.
pub trait Dijkstra: Graph
where
    Self::V: Ord + Clone,
    Self::W: Ord + Clone + Add<Self::W, Output = Self::W>,
{
    /// Calculate shortest path distance from `start` to all vertex.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O((E + V)logV).
    fn dijkstra<D>(&self, start: Self::V, init: Self::W, mut dist: D) -> D
    where
        D: DistanceTable<Self::V, Self::W>,
    {
        let mut pq = BinaryHeap::new();
        pq.push(Reverse((init.clone(), start.clone())));
        dist.set_distance(start, init);
        while let Some(Reverse((udist, u))) = pq.pop() {
            if dist.distance(&u).map(|d| *d < udist).unwrap_or(false) {
                continue;
            }
            for (v, uvdist) in self.childs(u) {
                let vdist = udist.clone() + uvdist;
                if dist.distance(&v).map(|d| *d < vdist).unwrap_or(false) {
                    continue;
                }
                dist.set_distance(v.clone(), vdist.clone());
                pq.push(Reverse((vdist, v)));
            }
        }
        dist
    }
}

impl<G: Graph> Dijkstra for G
where
    G::V: Ord + Clone,
    G::W: Ord + Clone + Add<G::W, Output = G::W>,
{
}

/// A trait to hold distance to any vertices.
///
/// This trait is for switch data structure by target vertices type.
pub trait DistanceTable<V, D> {
    fn distance(&self, v: &V) -> Option<&D>;
    fn set_distance(&mut self, v: V, d: D);
}

impl<D> DistanceTable<usize, D> for Vec<Option<D>> {
    fn distance(&self, v: &usize) -> Option<&D> {
        self[*v].as_ref()
    }

    fn set_distance(&mut self, v: usize, d: D) {
        self[v] = Some(d);
    }
}

impl<D> DistanceTable<(usize, usize), D> for Vec<Vec<Option<D>>> {
    fn distance(&self, v: &(usize, usize)) -> Option<&D> {
        self[v.0][v.1].as_ref()
    }

    fn set_distance(&mut self, v: (usize, usize), d: D) {
        self[v.0][v.1] = Some(d);
    }
}

impl<V: Hash + Eq, D> DistanceTable<V, D> for HashMap<V, D> {
    fn distance(&self, v: &V) -> Option<&D> {
        self.get(v)
    }

    fn set_distance(&mut self, v: V, d: D) {
        self.insert(v, d);
    }
}

impl<V: Ord, D> DistanceTable<V, D> for BTreeMap<V, D> {
    fn distance(&self, v: &V) -> Option<&D> {
        self.get(v)
    }

    fn set_distance(&mut self, v: V, d: D) {
        self.insert(v, d);
    }
}

#[cfg(test)]
mod test {
    use super::{Dijkstra, BFS};

    #[test]
    fn test_bfs() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ALDS1_11_C
        let graph = vec![vec![1, 3], vec![3], vec![], vec![2]];
        assert_eq!(
            graph.bfs(0, vec![None; 4]),
            vec![Some(0), Some(1), Some(2), Some(1)]
        );
    }

    #[test]
    fn test_dijkstra() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_A
        let graph = vec![
            vec![(1, 1), (2, 4)],
            vec![(2, 2), (3, 5)],
            vec![(3, 1)],
            vec![],
        ];
        assert_eq!(
            graph.dijkstra(0, 0, vec![None; 4]),
            vec![Some(0), Some(1), Some(3), Some(4)]
        );
    }
}
