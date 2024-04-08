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
    ///
    /// User should override default implementation if it is possible to return all edges directly.
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

/// An extension trait to add `dijkstra` which calculate single shortest path distance
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

/// An extension trait to add `bellman_ford` which calculate single shortest path distance
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

/// An extension trait to add `warshall_floyd` which calculate all pair shortest path distance
/// of the graph.
pub trait WarshallFloyd: Graph
where
    Self::W: PartialOrd + Add<Self::W, Output = Self::W>,
{
    /// Calculate shortest path distance of all vertices-vertices pair.
    ///
    /// This works even if this graph contains negative edge weight and
    /// if the graph contains negative cycle, return None.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// `is_negative` must return true if given weight is negative.
    ///
    /// Time complexity is O(V^3).
    fn warshall_floyd<D, F>(&self, init: Self::W, mut dist: D, mut is_negative: F) -> Option<D>
    where
        D: AllPairDistanceTable<Self::V, Self::W>,
        F: FnMut(&Self::W) -> bool,
    {
        let vs = self.vertex();
        for i in vs.iter().cloned() {
            for (j, w) in self.childs(i.clone()) {
                dist.set_distance(i.clone(), j, w);
            }
            dist.set_distance(i.clone(), i, init.clone());
        }
        for k in vs.iter().cloned() {
            for i in vs.iter().cloned() {
                for j in vs.iter().cloned() {
                    let (Some(dik), Some(dkj)) = (dist.distance(&i, &k), dist.distance(&k, &j))
                    else {
                        continue;
                    };
                    let new_dij = dik.clone() + dkj.clone();
                    if dist
                        .distance(&i, &j)
                        .map(|dij| *dij > new_dij)
                        .unwrap_or(true)
                    {
                        dist.set_distance(i.clone(), j, new_dij);
                    }
                }
            }
        }
        for i in vs {
            let Some(dii) = dist.distance(&i, &i) else {
                continue;
            };
            if is_negative(&dii) {
                return None;
            }
        }
        Some(dist)
    }
}

impl<G: Graph> WarshallFloyd for G where G::W: PartialOrd + Add<G::W, Output = G::W> {}

/// A trait to hold distance from any vertices to any vertices.
///
/// This trait is for switch data structure by target vertices type.
pub trait AllPairDistanceTable<V, D> {
    /// Get distance from `u` to `v`, or None if cannot go to `v` from `u`.
    fn distance(&self, u: &V, v: &V) -> Option<&D>;

    /// Set the distance of path from `u` to `v`.
    fn set_distance(&mut self, u: V, v: V, d: D);
}

impl<D> AllPairDistanceTable<usize, D> for Vec<Vec<Option<D>>> {
    fn distance(&self, u: &usize, v: &usize) -> Option<&D> {
        self[*u][*v].as_ref()
    }

    fn set_distance(&mut self, u: usize, v: usize, d: D) {
        self[u][v] = Some(d);
    }
}

impl<D> AllPairDistanceTable<(usize, usize), D> for Vec<Vec<Vec<Vec<Option<D>>>>> {
    fn distance(&self, u: &(usize, usize), v: &(usize, usize)) -> Option<&D> {
        self[u.0][u.1][v.0][v.1].as_ref()
    }

    fn set_distance(&mut self, u: (usize, usize), v: (usize, usize), d: D) {
        self[u.0][u.1][v.0][v.1] = Some(d);
    }
}

impl<V: Hash + Eq + Clone, D> AllPairDistanceTable<V, D> for HashMap<(V, V), D> {
    fn distance(&self, u: &V, v: &V) -> Option<&D> {
        self.get(&(u.clone(), v.clone()))
    }

    fn set_distance(&mut self, u: V, v: V, d: D) {
        self.insert((u, v), d);
    }
}

impl<V: Hash + Eq, D> AllPairDistanceTable<V, D> for HashMap<V, HashMap<V, D>> {
    fn distance(&self, u: &V, v: &V) -> Option<&D> {
        self.get(u)?.get(v)
    }

    fn set_distance(&mut self, u: V, v: V, d: D) {
        self.entry(u).or_default().insert(v, d);
    }
}

impl<V: Ord + Clone, D> AllPairDistanceTable<V, D> for BTreeMap<(V, V), D> {
    fn distance(&self, u: &V, v: &V) -> Option<&D> {
        self.get(&(u.clone(), v.clone()))
    }

    fn set_distance(&mut self, u: V, v: V, d: D) {
        self.insert((u, v), d);
    }
}

impl<V: Ord, D> AllPairDistanceTable<V, D> for BTreeMap<V, BTreeMap<V, D>> {
    fn distance(&self, u: &V, v: &V) -> Option<&D> {
        self.get(u)?.get(v)
    }

    fn set_distance(&mut self, u: V, v: V, d: D) {
        self.entry(u).or_default().insert(v, d);
    }
}

/// A trait to hold distance from a vertices to any vertices.
///
/// This trait is for switch data structure by target vertices type.
pub trait SingleSourceDistanceTable<V, D> {
    /// Get distance from a vertices to `v`, or None if cannot go to `v`.
    fn distance(&self, v: &V) -> Option<&D>;

    /// Set distance of path to `v`.
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
    use super::{BellmanFord, Dijkstra, WarshallFloyd, BFS};

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

    #[test]
    fn test_warshall_floyd() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_C
        let graph: Vec<Vec<(usize, i64)>> = vec![
            vec![(1, 1), (2, 5)],
            vec![(2, 2), (3, 4)],
            vec![(3, 1)],
            vec![(2, 7)],
        ];
        let dist = graph.warshall_floyd(0, vec![vec![None; 4]; 4], |w| *w < 0);
        assert_eq!(
            dist,
            Some(vec![
                vec![Some(0), Some(1), Some(3), Some(4)],
                vec![None, Some(0), Some(2), Some(3)],
                vec![None, None, Some(0), Some(1)],
                vec![None, None, Some(7), Some(0)],
            ])
        );
    }

    #[test]
    fn test_warshall_floyd_with_negative_cycle() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_C
        let graph: Vec<Vec<(usize, i64)>> = vec![
            vec![(1, 1), (2, 5)],
            vec![(2, 2), (3, 4)],
            vec![(3, 1)],
            vec![(2, -7)],
        ];
        assert_eq!(
            graph.warshall_floyd(0, vec![vec![None; 4]; 4], |w| *w < 0),
            None
        );
    }
}
