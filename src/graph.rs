use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::{Add, Range},
    slice::Iter,
};

/// A struct which represent a edge in graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge<V, W> {
    pub from: V,
    pub to: V,
    pub cost: W,
}

impl<V, W> Edge<V, W> {
    /// Construct a new `Edge` object.
    pub fn new(from: V, to: V, cost: W) -> Self {
        Self { from, to, cost }
    }

    pub fn as_tuple(&self) -> (&V, &V, &W) {
        (&self.from, &self.to, &self.cost)
    }
}

/// A struct which represent an adjacent vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Adjacent<V, W> {
    pub to: V,
    pub cost: W,
}

impl<V, W> Adjacent<V, W> {
    /// Construct a new `Adjacent` object.
    pub fn new(to: V, cost: W) -> Self {
        Self { to, cost }
    }

    pub fn as_tuple(&self) -> (&V, &W) {
        (&self.to, &self.cost)
    }
}

/// A trait represent directed/undirected graph with/without weight.
///
/// # Associated iterators
///
/// The `Vertices`, `Adjacents` and `Edges` requires that the type must be iterator and we can't write
/// such that `type Type = impl Iterator<..>` inside of impl block.
///
/// Usually, chained iterator has complicated type and some iterator method take closure so
/// that writing concrete type for the iterator is difficult or impossible.
///
/// One possible solution is define a new struct which hold initial iterator and implement
/// `Iterator` to it so that desired iterator is created from the initial iterator like below.
///
/// ```ignore
/// struct Wrapper<'a> {
///     iter: std::slice::Iter<'a, usize>
/// }
///
/// impl<'a> Iterator for Wrapper<'a> {
///     type Item = (usize, usize);
///     fn next(&self) -> Self::Item {
///         // Using map to process iterator's value.
///         self.iter.next().map(|v| (v, v))
///     }
/// }
/// ```
///
/// This method works well, but in competitive programming, this method is too slow for
/// implementing a custom `Graph`. So the next method stated below is better in such cases.
///
/// Next method is boxing the iterator like below.
///
/// ```ignore
/// impl Graph for GraphStruct {
///     // ...
///     type Vertices<'a> = Box<dyn Iterator<Item = Cow<'a, Self::V>> + 'a> where Self: 'a;
///     // ...
///
///     // ...
///     fn vertices(&self) -> Self::Vertices<'_> {
///         Box::new(todo!() /* write chained iterator here */)
///     }
/// }
/// ```
///
/// If we wrap iterator with `Box`, the type will be the style of `Box<dyn Iterator<Item = ...>>`
/// and it's easy to write. Also, this works even if the iterator using closure.
///
/// Boxing iterator take a cost, but if you can accept the cost, it's better.
pub trait Graph {
    type V: Clone;
    type W: Clone;
    type Vertices<'a>: Iterator<Item = Cow<'a, Self::V>>
    where
        Self: 'a;
    type Adjacents<'a>: Iterator<Item = Cow<'a, Adjacent<Self::V, Self::W>>>
    where
        Self: 'a;
    type Edges<'a>: Iterator<Item = Cow<'a, Edge<Self::V, Self::W>>>
    where
        Self: 'a;

    /// Iterate all vertex.
    fn vertices(&self) -> Self::Vertices<'_>;

    /// Returns number of vertex in this graph.
    ///
    /// By default, this call `vertices` and count the size of vector.
    ///
    /// User should override default implementation if it is possible to return the number of
    /// vertex directly.
    fn vertex_count(&self) -> usize {
        self.vertices().count()
    }

    /// Iterate all adjacent of specified vertex.
    fn adjacents(&self, v: Self::V) -> Self::Adjacents<'_>;

    /// Iterate all edge.
    fn edges(&self) -> Self::Edges<'_>;
}

pub struct AdjacentList<W> {
    adjs: Vec<Vec<Adjacent<usize, W>>>,
    edges: Vec<Edge<usize, W>>,
}

impl<W: Clone> AdjacentList<W> {
    /// Construct a new `AdjacentList`.
    pub fn new(n: usize) -> Self {
        Self {
            adjs: vec![Vec::new(); n],
            edges: Vec::new(),
        }
    }

    /// Add an edge to graph.
    pub fn add_edge(&mut self, edge: Edge<usize, W>) {
        self.adjs[edge.from].push(Adjacent::new(edge.to, edge.cost.clone()));
        self.edges.push(edge);
    }
}

pub struct AdjacentListVertices<'a> {
    iter: Range<usize>,
    _phantom: PhantomData<&'a usize>,
}

impl<'a> AdjacentListVertices<'a> {
    fn new(iter: Range<usize>) -> Self {
        Self {
            iter,
            _phantom: PhantomData,
        }
    }
}

impl<'a> Iterator for AdjacentListVertices<'a> {
    type Item = Cow<'a, usize>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|v| Cow::Owned(v))
    }
}

pub struct AdjacentListEdges<'a, W> {
    iter: Iter<'a, Edge<usize, W>>,
}

impl<'a, W> AdjacentListEdges<'a, W> {
    fn new(iter: Iter<'a, Edge<usize, W>>) -> Self {
        Self { iter }
    }
}

impl<'a, W: Clone> Iterator for AdjacentListEdges<'a, W> {
    type Item = Cow<'a, Edge<usize, W>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|edge| Cow::Borrowed(edge))
    }
}

pub struct AdjacentListAdjacents<'a, W> {
    iter: Iter<'a, Adjacent<usize, W>>,
}

impl<'a, W> AdjacentListAdjacents<'a, W> {
    fn new(iter: Iter<'a, Adjacent<usize, W>>) -> Self {
        Self { iter }
    }
}

impl<'a, W: Clone> Iterator for AdjacentListAdjacents<'a, W> {
    type Item = Cow<'a, Adjacent<usize, W>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|adj| Cow::Borrowed(adj))
    }
}

impl<W: Clone> Graph for AdjacentList<W> {
    type V = usize;
    type W = W;
    type Vertices<'a> = AdjacentListVertices<'a> where Self: 'a;
    type Adjacents<'a> = AdjacentListAdjacents<'a, W> where Self: 'a;
    type Edges<'a> = AdjacentListEdges<'a, W> where Self: 'a;

    fn vertices(&self) -> Self::Vertices<'_> {
        AdjacentListVertices::new(0..self.adjs.len())
    }

    fn vertex_count(&self) -> usize {
        self.adjs.len()
    }

    fn adjacents(&self, v: Self::V) -> Self::Adjacents<'_> {
        AdjacentListAdjacents::new(self.adjs[v].iter())
    }

    fn edges(&self) -> Self::Edges<'_> {
        AdjacentListEdges::new(self.edges.iter())
    }
}

/// An extension trait to add `bfs` to travel the graph and report if the vertex is visitable
/// from specified vertex.
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
            for adj in self.adjacents(u.clone()) {
                let (v, _) = adj.as_tuple();
                if dist.distance(v).is_none() {
                    dist.set_distance(v.clone(), dist.distance(&u).unwrap() + 1);
                    queue.push_back(v.clone());
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
            for adj in self.adjacents(u) {
                let (v, uvdist) = adj.as_tuple();
                let vdist = udist.clone() + uvdist.clone();
                if dist.distance(&v).map(|d| *d < vdist).unwrap_or(false) {
                    continue;
                }
                dist.set_distance(v.clone(), vdist.clone());
                pq.push(Reverse((vdist, v.clone())));
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
        let n = self.vertex_count();
        dist.set_distance(start, init);
        for i in 0..n {
            for edge in self.edges() {
                let (u, v, uvdist) = edge.as_tuple();
                let Some(udist) = dist.distance(u) else {
                    continue;
                };
                let next_vdist = udist.clone() + uvdist.clone();
                if dist
                    .distance(v)
                    .map(|vdist| next_vdist < *vdist)
                    .unwrap_or(true)
                {
                    if i == n - 1 {
                        return None;
                    }
                    dist.set_distance(v.clone(), next_vdist);
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
    /// Calculate shortest path distance of all vertex-vertex pair.
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
        for i in self.vertices() {
            for adj in self.adjacents((*i).clone()) {
                let (j, w) = adj.as_tuple();
                dist.set_distance((*i).clone(), j.clone(), w.clone());
            }
            dist.set_distance((*i).clone(), (*i).clone(), init.clone());
        }
        for k in self.vertices() {
            for i in self.vertices() {
                for j in self.vertices() {
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
                        dist.set_distance((*i).clone(), (*j).clone(), new_dij);
                    }
                }
            }
        }
        for i in self.vertices() {
            let Some(dii) = dist.distance(&i, &i) else {
                continue;
            };
            if is_negative(dii) {
                return None;
            }
        }
        Some(dist)
    }
}

impl<G: Graph> WarshallFloyd for G where G::W: PartialOrd + Add<G::W, Output = G::W> {}

/// A trait to hold distance from any vertex to any vertex.
///
/// This trait is for switch data structure by target vertex type.
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

// TODO: Remove Clone from bounds?
impl<V: Hash + Eq + Clone, D> AllPairDistanceTable<V, D> for HashMap<(V, V), D> {
    fn distance(&self, u: &V, v: &V) -> Option<&D> {
        self.get(&(u.clone(), v.clone()))
    }

    fn set_distance(&mut self, u: V, v: V, d: D) {
        self.insert((u, v), d);
    }
}

// TODO: Remove Clone from bounds?
impl<V: Ord + Clone, D> AllPairDistanceTable<V, D> for BTreeMap<(V, V), D> {
    fn distance(&self, u: &V, v: &V) -> Option<&D> {
        self.get(&(u.clone(), v.clone()))
    }

    fn set_distance(&mut self, u: V, v: V, d: D) {
        self.insert((u, v), d);
    }
}

/// A trait to hold distance from a vertexto any vertex.
///
/// This trait is for switch data structure by target vertex type.
pub trait SingleSourceDistanceTable<V, D> {
    /// Get distance from a vertex to `v`, or None if cannot go to `v`.
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
    use crate::graph::Edge;

    use super::{AdjacentList, BellmanFord, Dijkstra, WarshallFloyd, BFS};

    #[test]
    fn test_bfs() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ALDS1_11_C
        let mut graph = AdjacentList::new(4);
        graph.add_edge(Edge::new(0, 1, ()));
        graph.add_edge(Edge::new(0, 3, ()));
        graph.add_edge(Edge::new(1, 3, ()));
        graph.add_edge(Edge::new(3, 2, ()));
        assert_eq!(
            graph.bfs(0, vec![None; 4]),
            vec![Some(0), Some(1), Some(2), Some(1)]
        );
    }

    #[test]
    fn test_dijkstra() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_A
        let mut graph = AdjacentList::new(4);
        graph.add_edge(Edge::new(0, 1, 1));
        graph.add_edge(Edge::new(0, 2, 4));
        graph.add_edge(Edge::new(1, 2, 2));
        graph.add_edge(Edge::new(2, 3, 1));
        graph.add_edge(Edge::new(1, 3, 5));
        assert_eq!(
            graph.dijkstra(0, 0, vec![None; 4]),
            vec![Some(0), Some(1), Some(3), Some(4)]
        );
    }

    #[test]
    fn test_bellman_ford() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_B
        let mut graph = AdjacentList::new(4);
        graph.add_edge(Edge::new(0, 1, 2));
        graph.add_edge(Edge::new(0, 2, 3));
        graph.add_edge(Edge::new(1, 2, -5));
        graph.add_edge(Edge::new(1, 3, 1));
        graph.add_edge(Edge::new(2, 3, 2));
        assert_eq!(
            graph.bellman_ford(0, 0, vec![None; 4]),
            Some(vec![Some(0), Some(2), Some(-3), Some(-1)])
        );
    }

    #[test]
    fn test_bellman_ford_with_negative_cycle() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_B
        let mut graph = AdjacentList::new(4);
        graph.add_edge(Edge::new(0, 1, 2));
        graph.add_edge(Edge::new(0, 2, 3));
        graph.add_edge(Edge::new(1, 2, -5));
        graph.add_edge(Edge::new(1, 3, 1));
        graph.add_edge(Edge::new(2, 3, 2));
        graph.add_edge(Edge::new(3, 1, 0));
        assert_eq!(graph.bellman_ford(0, 0, vec![None; 4]), None);
    }

    #[test]
    fn test_warshall_floyd() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_C
        let mut graph = AdjacentList::new(4);
        graph.add_edge(Edge::new(0, 1, 1));
        graph.add_edge(Edge::new(0, 2, 5));
        graph.add_edge(Edge::new(1, 2, 2));
        graph.add_edge(Edge::new(1, 3, 4));
        graph.add_edge(Edge::new(2, 3, 1));
        graph.add_edge(Edge::new(3, 2, 7));
        assert_eq!(
            graph.warshall_floyd(0, vec![vec![None; 4]; 4], |w| *w < 0),
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
        let mut graph = AdjacentList::new(4);
        graph.add_edge(Edge::new(0, 1, 1));
        graph.add_edge(Edge::new(0, 2, 5));
        graph.add_edge(Edge::new(1, 2, 2));
        graph.add_edge(Edge::new(1, 3, 4));
        graph.add_edge(Edge::new(2, 3, 1));
        graph.add_edge(Edge::new(3, 2, -7));
        assert_eq!(
            graph.warshall_floyd(0, vec![vec![None; 4]; 4], |w| *w < 0),
            None
        );
    }
}
