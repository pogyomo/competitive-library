use std::iter::{successors, zip};

/// Calculate x^n mod p.
fn pow_mod(x: u64, mut n: u64, p: u64) -> u64 {
    assert!(p >= 2);
    let mut xn = x;
    let mut res = 1;
    while n != 0 {
        if n & 1 != 0 {
            res = (res * xn) % p;
        }
        xn = (xn * xn) % p;
        n >>= 1;
    }
    res
}

/// Calculate x which satisfy xn == 1 mod p.
fn inv_mod(n: u64, p: u64) -> u64 {
    assert!(p >= 2);
    pow_mod(n, p - 2, p)
}

/// Find primitive root r such that smallest m which satisfy r^m == 1 mod p is p - 1.
fn find_primitive_root(p: u64) -> u64 {
    // reference: https://37zigen.com/primitive-root
    let ms = {
        let mut res = Vec::new();
        let mut n = p - 1;
        let mut k = 2;
        while k * k <= n {
            if n % k == 0 {
                res.push(k);
                while n % k == 0 {
                    n /= k;
                }
            }
            k += 1;
        }
        res
    };
    let mut g = 2;
    while ms.iter().copied().any(|m| pow_mod(g, (p - 1) / m, p) == 1) {
        g += 1;
    }
    g
}

/// A struct for number theoretic transform and related operation.
pub struct NTT {
    p: u64,
    root: Vec<u64>,
    iroot: Vec<u64>,
}

impl NTT {
    /// Construct a new NTT object. `p` must be a prime.
    pub fn new(p: u64) -> Self {
        // Find a and m such that p = a * 2^m + 1
        let (a, m) = {
            let mut m = 0;
            let mut p = p - 1;
            while p & 1 == 0 {
                m += 1;
                p >>= 1;
            }
            (p, m)
        };
        let r = pow_mod(find_primitive_root(p), a, p);
        let mut root = successors(Some(r), |r| Some((r * r) % p))
            .take(m)
            .collect::<Vec<_>>();
        root.reverse();
        let iroot = root.clone().into_iter().map(|v| inv_mod(v, p)).collect();
        Self { p, root, iroot }
    }

    /// Construct a new NTT object with `p` = 998244353.
    pub fn new998244353() -> Self {
        Self::new(998244353)
    }

    /// Perform number theoretic transform. We expect the size of `a` is power of two.
    ///
    /// Time complexity is O(nlogn).
    pub fn ntt(&self, a: Vec<u64>) -> Vec<u64> {
        self.ntt_impl(a, false)
    }

    /// Perform inverse number theoretic transform. We expect the size of `a` is power of two.
    ///
    /// Time complexity is O(nlogn).
    pub fn intt(&self, a: Vec<u64>) -> Vec<u64> {
        let n = a.len() as u64;
        self.ntt_impl(a, true)
            .into_iter()
            .map(|v| (v * inv_mod(n, self.p)) % self.p)
            .collect()
    }

    /// Calculate convolution of given two sequences.
    ///
    /// If at least one of two sequences is empty, return empty sequence.
    ///
    /// Time complexity is O(nlogn).
    pub fn convolution(&self, mut a: Vec<u64>, mut b: Vec<u64>) -> Vec<u64> {
        if a.len() == 0 || b.len() == 0 {
            return Vec::new();
        }

        let res_n = a.len() + b.len() - 1;
        let n = res_n.next_power_of_two();
        a.resize(n, 0);
        b.resize(n, 0);
        let da = self.ntt(a);
        let db = self.ntt(b);
        let dres = zip(da, db).map(|(a, b)| (a * b) % self.p).collect();
        self.intt(dres).into_iter().take(res_n).collect()
    }

    fn ntt_impl(&self, a: Vec<u64>, inv: bool) -> Vec<u64> {
        let n = a.len();
        if n == 1 {
            return a;
        }
        let mut f = Vec::with_capacity(n >> 1);
        let mut g = Vec::with_capacity(n >> 1);
        for (i, a) in a.into_iter().enumerate() {
            if i & 1 == 0 {
                f.push(a);
            } else {
                g.push(a);
            }
        }
        let f = self.ntt_impl(f, inv);
        let g = self.ntt_impl(g, inv);
        let k = n.trailing_zeros() as usize - 1;
        let r = if inv { self.iroot[k] } else { self.root[k] };
        let mut m = 1;
        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            res.push((f[i % (n >> 1)] + (m * g[i % (n >> 1)]) % self.p) % self.p);
            m = (m * r) % self.p;
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::{inv_mod, pow_mod, NTT};

    #[test]
    fn test_pow_mod() {
        let p = 998244353;
        assert_eq!(pow_mod(10, p - 1, p), 1);
    }

    #[test]
    fn test_inv_mod() {
        let p = 998244353;
        assert_eq!((inv_mod(10, p) * 10) % p, 1);
        assert_eq!(pow_mod(10, p - 1, p), 1);
    }

    #[test]
    fn test_ntt_and_intt() {
        let v = vec![1, 2, 3, 4];
        let ntt = NTT::new998244353();
        assert_eq!(ntt.intt(ntt.ntt(v.clone())), v);
    }

    #[test]
    fn test_convolution() {
        let ntt = NTT::new998244353();
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        let mut ab = vec![0; a.len() + b.len() - 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                ab[i + j] += a[i] * b[j];
            }
        }
        assert_eq!(ntt.convolution(a, b), ab);
    }

    #[test]
    fn test_convolution_with_empty_list() {
        let ntt = NTT::new998244353();
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        assert_eq!(ntt.convolution(a, Vec::new()), Vec::new());
        assert_eq!(ntt.convolution(Vec::new(), b), Vec::new());
        assert_eq!(ntt.convolution(Vec::new(), Vec::new()), Vec::new());
    }
}
