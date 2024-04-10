use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::Add,
};

/// A trait represent adjacent of a vertex.
pub trait Adjacent<V, W> {
    /// Returns adjacent vertex.
    fn to(&self) -> &V;

    /// Returns cost to go to the adjacent vertex.
    fn cost(&self) -> &W;
}

impl<V, W> Adjacent<V, W> for (V, W) {
    fn to(&self) -> &V {
        &self.0
    }

    fn cost(&self) -> &W {
        &self.1
    }
}

/// A trait represent edge in graph.
pub trait Edge<V, W> {
    /// Returns vertex goes into this edge.
    fn from(&self) -> &V;

    /// Returns vertex comes out this edge.
    fn to(&self) -> &V;

    /// Returns the weight of this edge.
    fn cost(&self) -> &W;
}

impl<V, W> Edge<V, W> for (V, V, W) {
    fn from(&self) -> &V {
        &self.0
    }

    fn to(&self) -> &V {
        &self.1
    }

    fn cost(&self) -> &W {
        &self.2
    }
}

/// A graph which can enumerate all vertices in graph.
pub trait VertexEnumeratableGraph<V: Clone> {
    /// The kind of iterator to be used to iterate the vertices.
    type Vertices<'a>: Iterator<Item = Cow<'a, V>>
    where
        V: 'a,
        Self: 'a;

    /// Returns iterator over all vertices in this graph.
    fn vertices(&self) -> Self::Vertices<'_>;
}

/// A graph which can count all vertices in graph.
pub trait VertexCountableGraph<V: Clone>: VertexEnumeratableGraph<V> {
    /// Returns number of vertex in this graph.
    fn vertex_count(&self) -> usize {
        self.vertices().count()
    }
}

/// A graph which can enumerate all adjacents of the vertex.
pub trait AdjacentEnumeratableGraph<V, W> {
    /// The type of the adjacents to be enumerated.
    type Adjacent: Adjacent<V, W> + Clone;

    /// The kind of iterator to be used to iterate the adjacents.
    type Adjacents<'a>: Iterator<Item = Cow<'a, Self::Adjacent>>
    where
        Self::Adjacent: 'a,
        Self: 'a;

    /// Returns iterator over all adjacents of the vertex `v`.
    fn adjacents(&self, v: V) -> Self::Adjacents<'_>;
}

/// A graph which can enumerate all edges in the graph.
pub trait EdgeEnumeratableGraph<V, W> {
    /// The type of edge to be enumerated.
    type Edge: Edge<V, W> + Clone;

    /// The kind of iterator to be used to iterate the edges.
    type Edges<'a>: Iterator<Item = Cow<'a, Self::Edge>>
    where
        Self::Edge: 'a,
        Self: 'a;

    /// Returns iterator over all edges in this graph.
    fn edges(&self) -> Self::Edges<'_>;
}

/// A simple adjacent list where vertex type is `usize`.
pub struct AdjacentList<W> {
    adjs: Vec<Vec<(usize, W)>>,
    edges: Vec<(usize, usize, W)>,
}

impl<W: Clone> AdjacentList<W> {
    /// Returns a new `AdjacentList` with the number of vertex is `n`.
    pub fn new(n: usize) -> Self {
        Self {
            adjs: vec![Vec::new(); n],
            edges: Vec::new(),
        }
    }

    /// Add edge to this list.
    pub fn add_edge(&mut self, from: usize, to: usize, cost: W) {
        self.adjs[from].push((to, cost.clone()));
        self.edges.push((from, to, cost));
    }
}

impl<W> VertexEnumeratableGraph<usize> for AdjacentList<W> {
    type Vertices<'a> = AdjacentListVertices<'a> where W: 'a;

    fn vertices(&self) -> Self::Vertices<'_> {
        AdjacentListVertices {
            iter: 0..self.adjs.len(),
            _phantom: PhantomData,
        }
    }
}

impl<W> VertexCountableGraph<usize> for AdjacentList<W> {
    fn vertex_count(&self) -> usize {
        self.adjs.len()
    }
}

impl<W: Clone> AdjacentEnumeratableGraph<usize, W> for AdjacentList<W> {
    type Adjacent = (usize, W);
    type Adjacents<'a> = AdjacentListAdjacents<'a, W> where W: 'a;

    fn adjacents(&self, v: usize) -> Self::Adjacents<'_> {
        AdjacentListAdjacents {
            iter: self.adjs[v].iter(),
        }
    }
}

impl<W: Clone> EdgeEnumeratableGraph<usize, W> for AdjacentList<W> {
    type Edge = (usize, usize, W);
    type Edges<'a> = AdjacentListEdges<'a, W> where W: 'a;

    fn edges(&self) -> Self::Edges<'_> {
        AdjacentListEdges {
            iter: self.edges.iter(),
        }
    }
}

/// An iterator over all vertices of `AdjacentList`.
pub struct AdjacentListVertices<'a> {
    iter: std::ops::Range<usize>,
    _phantom: PhantomData<&'a usize>,
}

impl<'a> Iterator for AdjacentListVertices<'a> {
    type Item = Cow<'a, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|v| Cow::Owned(v))
    }
}

/// An iterator over all adjacents of a vertex of `AdjacentList`.
pub struct AdjacentListAdjacents<'a, W> {
    iter: std::slice::Iter<'a, (usize, W)>,
}

impl<'a, W: Clone> Iterator for AdjacentListAdjacents<'a, W> {
    type Item = Cow<'a, (usize, W)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|v| Cow::Borrowed(v))
    }
}

/// An iterator over all edges of `AdjacentList`.
pub struct AdjacentListEdges<'a, W> {
    iter: std::slice::Iter<'a, (usize, usize, W)>,
}

impl<'a, W: Clone> Iterator for AdjacentListEdges<'a, W> {
    type Item = Cow<'a, (usize, usize, W)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|edge| Cow::Borrowed(edge))
    }
}

/// An extension trait to add `bfs` to travel the graph and report if the vertex is visitable
/// from specified vertex.
pub trait BFS<V, W>: AdjacentEnumeratableGraph<V, W>
where
    V: Clone,
{
    /// Calculate minimum number of edge in path from `start` to all vertex.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O(V + E).
    fn bfs<D>(&self, start: V, mut dist: D) -> D
    where
        D: SingleSourceDistanceTable<V, usize>,
    {
        let mut queue = VecDeque::new();
        queue.push_back(start.clone());
        dist.set_distance(start, 0);
        while let Some(u) = queue.pop_front() {
            for adj in self.adjacents(u.clone()) {
                let v = adj.to();
                if dist.distance(v).is_none() {
                    dist.set_distance(v.clone(), dist.distance(&u).unwrap() + 1);
                    queue.push_back(v.clone());
                }
            }
        }
        dist
    }
}

impl<V, W, G> BFS<V, W> for G
where
    G: AdjacentEnumeratableGraph<V, W>,
    V: Clone,
{
}

/// An extension trait to add `dijkstra` which calculate single shortest path distance
/// of the graph.
pub trait Dijkstra<V, W>: AdjacentEnumeratableGraph<V, W>
where
    V: Ord + Clone,
    W: Ord + Clone + Add<W, Output = W>,
{
    /// Calculate shortest path distance from `start` to all vertex.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O((E + V)logV).
    fn dijkstra<D>(&self, start: V, init: W, mut dist: D) -> D
    where
        D: SingleSourceDistanceTable<V, W>,
    {
        let mut pq = BinaryHeap::new();
        pq.push(Reverse((init.clone(), start.clone())));
        dist.set_distance(start, init);
        while let Some(Reverse((udist, u))) = pq.pop() {
            if dist.distance(&u).map(|d| *d < udist).unwrap_or(false) {
                continue;
            }
            for adj in self.adjacents(u) {
                let (v, uvdist) = (adj.to(), adj.cost());
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

impl<V, W, G> Dijkstra<V, W> for G
where
    G: AdjacentEnumeratableGraph<V, W>,
    V: Ord + Clone,
    W: Ord + Clone + Add<W, Output = W>,
{
}

/// An extension trait to add `bellman_ford` which calculate single shortest path distance
/// of the graph.
pub trait BellmanFord<V, W>: EdgeEnumeratableGraph<V, W> + VertexCountableGraph<V>
where
    V: Clone,
    W: PartialOrd + Clone + Add<W, Output = W>,
{
    /// Calculate shortest path distance from `start` to all vertex.
    ///
    /// This works even if this graph contains negative edge weight and
    /// if the graph contains negative cycle, return None.
    ///
    /// `dist` must be initialized to None for all vertex.
    ///
    /// Time complexity is O(VE).
    fn bellman_ford<D>(&self, start: V, init: W, mut dist: D) -> Option<D>
    where
        D: SingleSourceDistanceTable<V, W>,
    {
        let n = self.vertex_count();
        dist.set_distance(start, init);
        for i in 0..n {
            for edge in self.edges() {
                let (u, v, uvdist) = (edge.from(), edge.to(), edge.cost());
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

impl<V, W, G> BellmanFord<V, W> for G
where
    G: EdgeEnumeratableGraph<V, W> + VertexCountableGraph<V>,
    V: Clone,
    W: PartialOrd + Clone + Add<W, Output = W>,
{
}

/// An extension trait to add `warshall_floyd` which calculate all pair shortest path distance
/// of the graph.
pub trait WarshallFloyd<V, W>: VertexCountableGraph<V> + AdjacentEnumeratableGraph<V, W>
where
    V: Clone,
    W: PartialOrd + Clone + Add<W, Output = W>,
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
    fn warshall_floyd<D, F>(&self, init: W, mut dist: D, mut is_negative: F) -> Option<D>
    where
        D: AllPairDistanceTable<V, W>,
        F: FnMut(&W) -> bool,
    {
        for i in self.vertices() {
            for adj in self.adjacents((*i).clone()) {
                let (j, w) = (adj.to(), adj.cost());
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

impl<V, W, G> WarshallFloyd<V, W> for G
where
    G: VertexCountableGraph<V> + AdjacentEnumeratableGraph<V, W>,
    V: Clone,
    W: PartialOrd + Clone + Add<W, Output = W>,
{
}

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
    use super::{AdjacentList, BellmanFord, Dijkstra, WarshallFloyd, BFS};

    #[test]
    fn test_bfs() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ALDS1_11_C
        let mut graph = AdjacentList::new(4);
        graph.add_edge(0, 1, ());
        graph.add_edge(0, 3, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(3, 2, ());
        assert_eq!(
            graph.bfs(0, vec![None; 4]),
            vec![Some(0), Some(1), Some(2), Some(1)]
        );
    }

    #[test]
    fn test_dijkstra() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_A
        let mut graph = AdjacentList::new(4);
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 4);
        graph.add_edge(1, 2, 2);
        graph.add_edge(2, 3, 1);
        graph.add_edge(1, 3, 5);
        assert_eq!(
            graph.dijkstra(0, 0, vec![None; 4]),
            vec![Some(0), Some(1), Some(3), Some(4)]
        );
    }

    #[test]
    fn test_bellman_ford() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_B
        let mut graph = AdjacentList::new(4);
        graph.add_edge(0, 1, 2);
        graph.add_edge(0, 2, 3);
        graph.add_edge(1, 2, -5);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 3, 2);
        assert_eq!(
            graph.bellman_ford(0, 0, vec![None; 4]),
            Some(vec![Some(0), Some(2), Some(-3), Some(-1)])
        );
    }

    #[test]
    fn test_bellman_ford_with_negative_cycle() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_B
        let mut graph = AdjacentList::new(4);
        graph.add_edge(0, 1, 2);
        graph.add_edge(0, 2, 3);
        graph.add_edge(1, 2, -5);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 3, 2);
        graph.add_edge(3, 1, 0);
        assert_eq!(graph.bellman_ford(0, 0, vec![None; 4]), None);
    }

    #[test]
    fn test_warshall_floyd() {
        // test case come from https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_C
        let mut graph = AdjacentList::new(4);
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 5);
        graph.add_edge(1, 2, 2);
        graph.add_edge(1, 3, 4);
        graph.add_edge(2, 3, 1);
        graph.add_edge(3, 2, 7);
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
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 5);
        graph.add_edge(1, 2, 2);
        graph.add_edge(1, 3, 4);
        graph.add_edge(2, 3, 1);
        graph.add_edge(3, 2, -7);
        assert_eq!(
            graph.warshall_floyd(0, vec![vec![None; 4]; 4], |w| *w < 0),
            None
        );
    }
}
