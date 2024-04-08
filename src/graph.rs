use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, VecDeque},
    hash::Hash,
    ops::Add,
};

/// A trait represent directed/undirected graph with/without weight.
pub trait Graph {
    type V: Clone;
    type W: Clone;
    /// Collect all vertex of this graph.
    fn vertex(&self) -> Vec<Self::V>;

    /// Returns number of vertex in this graph.
    fn vertex_count(&self) -> usize;

    /// Collect all childs of specified vertices.
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)>;

    /// Collect all edges of this graph.
    ///
    /// By default, this call `childs` for all vertex gained from `vertex`
    /// to collect all edges.
    /// User should override default implementation if the number of vertex is large but the number
    /// of edge is small.
    fn edges(&self) -> Vec<(Self::V, Self::V, Self::W)> {
        let mut res = Vec::new();
        for u in self.vertex() {
            for (v, w) in self.childs(u.clone()) {
                res.push((u.clone(), v, w));
            }
        }
        res
    }
}

impl Graph for Vec<Vec<usize>> {
    type V = usize;
    type W = ();
    fn vertex(&self) -> Vec<Self::V> {
        (0..self.len()).collect()
    }
    fn vertex_count(&self) -> usize {
        self.len()
    }
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self[v].iter().copied().map(|v| (v, ())).collect()
    }
}

impl<W: Clone> Graph for Vec<Vec<(usize, W)>> {
    type V = usize;
    type W = W;
    fn vertex(&self) -> Vec<Self::V> {
        (0..self.len()).collect()
    }
    fn vertex_count(&self) -> usize {
        self.len()
    }
    fn childs(&self, v: Self::V) -> Vec<(Self::V, Self::W)> {
        self[v].clone()
    }
}

/// An extension trait to add `bfs` to travel the graph and report if the vertices is visitable
/// from specified vertices.
pub trait BFS: Graph {
    /// Calculate minimum number of edge in path from `start` to all vertex.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O(V + E).
    fn bfs<D>(&self, start: Self::V, mut dist: D) -> D
    where
        D: SingleSourceDistanceTable<Self::V, usize>,
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

impl<G: Graph> BFS for G {}

/// An extension trait to add `dijkstra` which calculate single shortest path distance with
/// of the graph.
pub trait Dijkstra: Graph
where
    Self::V: Ord,
    Self::W: Ord + Add<Self::W, Output = Self::W>,
{
    /// Calculate shortest path distance from `start` to all vertex.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O((E + V)logV).
    fn dijkstra<D>(&self, start: Self::V, init: Self::W, mut dist: D) -> D
    where
        D: SingleSourceDistanceTable<Self::V, Self::W>,
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
    G::V: Ord,
    G::W: Ord + Add<G::W, Output = G::W>,
{
}

/// An extension trait to add `bellman_ford` which calculate single shortest path distance with
/// of the graph.
pub trait BellmanFord: Graph
where
    Self::W: PartialOrd + Add<Self::W, Output = Self::W>,
{
    /// Calculate shortest path distance from `start` to all vertex.
    ///
    /// This works even if this graph contains negative edge weight and
    /// if the graph contains negative cycle, return None.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O(VE).
    fn bellman_ford<D>(&self, start: Self::V, init: Self::W, mut dist: D) -> Option<D>
    where
        D: SingleSourceDistanceTable<Self::V, Self::W>,
    {
        // TODO: test if the algorithm is correct
        let vcount = self.vertex_count();
        let edges = self.edges();
        dist.set_distance(start, init);
        for i in 0..vcount {
            for (u, v, w) in edges.iter().cloned() {
                let Some(udist) = dist.distance(&u) else {
                    continue;
                };
                let next_vdist = udist.clone() + w;
                if dist
                    .distance(&v)
                    .map(|vdist| next_vdist < *vdist)
                    .unwrap_or(true)
                {
                    if i == vcount - 1 {
                        return None;
                    }
                    dist.set_distance(v, next_vdist);
                }
            }
        }
        Some(dist)
    }
}

impl<G: Graph> BellmanFord for G where G::W: PartialOrd + Add<G::W, Output = G::W> {}

/// A trait to hold distance from a vertices to any vertices.
///
/// This trait is for switch data structure by target vertices type.
pub trait SingleSourceDistanceTable<V, D> {
    fn distance(&self, v: &V) -> Option<&D>;
    fn set_distance(&mut self, v: V, d: D);
}

impl<D> SingleSourceDistanceTable<usize, D> for Vec<Option<D>> {
    fn distance(&self, v: &usize) -> Option<&D> {
        self[*v].as_ref()
    }

    fn set_distance(&mut self, v: usize, d: D) {
        self[v] = Some(d);
    }
}

impl<D> SingleSourceDistanceTable<(usize, usize), D> for Vec<Vec<Option<D>>> {
    fn distance(&self, v: &(usize, usize)) -> Option<&D> {
        self[v.0][v.1].as_ref()
    }

    fn set_distance(&mut self, v: (usize, usize), d: D) {
        self[v.0][v.1] = Some(d);
    }
}

impl<V: Hash + Eq, D> SingleSourceDistanceTable<V, D> for HashMap<V, D> {
    fn distance(&self, v: &V) -> Option<&D> {
        self.get(v)
    }

    fn set_distance(&mut self, v: V, d: D) {
        self.insert(v, d);
    }
}

impl<V: Ord, D> SingleSourceDistanceTable<V, D> for BTreeMap<V, D> {
    fn distance(&self, v: &V) -> Option<&D> {
        self.get(v)
    }

    fn set_distance(&mut self, v: V, d: D) {
        self.insert(v, d);
    }
}

#[cfg(test)]
mod test {
    use super::{BellmanFord, Dijkstra, BFS};

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

    #[test]
    fn test_bellman_ford() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_B
        let graph: Vec<Vec<(usize, i64)>> = vec![
            vec![(1, 2), (2, 3)],
            vec![(2, -5), (3, 1)],
            vec![(3, 2)],
            vec![],
        ];
        assert_eq!(
            graph.bellman_ford(0, 0, vec![None; 4]),
            Some(vec![Some(0), Some(2), Some(-3), Some(-1)])
        );
    }

    #[test]
    fn test_bellman_ford_with_negative_cycle() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_B
        let graph: Vec<Vec<(usize, i64)>> = vec![
            vec![(1, 2), (2, 3)],
            vec![(2, -5), (3, 1)],
            vec![(3, 2)],
            vec![(1, 0)],
        ];
        assert_eq!(graph.bellman_ford(0, 0, vec![None; 4]), None);
    }
}
