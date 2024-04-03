use std::{collections::BTreeMap, iter::successors};

// Find great common divider of `a` and `b`. Time complexity is O(logn)
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

// Find a prime factor of `n` using Pollard's rho algorithm. We expect given `n` is not prime.
// Time complexity is O(n^(1/4)polylog(n)).
fn find_prime_factor(n: u64) -> u64 {
    // reference: https://qiita.com/t_fuki/items/7cd50de54d3c5d063b4a
    if n & 1 == 0 {
        return 2;
    }
    for c in 1..n {
        let f = |x: u64| (((x as u128 * x as u128) % n as u128) as u64 + c) % n;
        let (mut x, mut y) = (0, 0);
        let mut g = 1;
        while g == 1 {
            x = f(x);
            y = f(f(y));
            g = gcd(x.abs_diff(y), n);
        }
        if g == n {
            continue;
        }
        if is_prime(g) {
            return g;
        } else if is_prime(n / g) {
            return n / g;
        } else {
            return find_prime_factor(g);
        }
    }
    unreachable!()
}

/// Find all prime factor and its amount. Time complexity is O(n^(1/4)polylog(n)).
pub fn prime_factorize(mut n: u64) -> Vec<(u64, usize)> {
    let mut res = BTreeMap::new();
    while n > 1 && !is_prime(n) {
        let p = find_prime_factor(n);
        let mut count = 0;
        while n % p == 0 {
            n /= p;
            count += 1;
        }
        res.insert(p, count);
    }
    if n > 1 {
        res.insert(n, 1);
    }
    res.into_iter().collect()
}

/// A struct to perform prime factorize by using pre-calculated table.
#[derive(Clone)]
pub struct PrimeFactorizer {
    spf: Vec<usize>,
}

// reference: https://algo-logic.info/prime-fact
impl PrimeFactorizer {
    /// Prepare for multiple prime factorization query. Time complexity is O(nloglogn).
    pub fn new(n: usize) -> Self {
        let mut spf = Vec::with_capacity(n + 1);
        for i in 0..=n {
            spf.push(i);
        }
        for i in (2..).take_while(|i| i * i <= n) {
            if spf[i] == i {
                for j in successors(Some(i), |j| Some(j + i)).take_while(|&j| j <= n) {
                    if spf[j] == j {
                        spf[j] = i;
                    }
                }
            }
        }
        PrimeFactorizer { spf }
    }

    /// Find all prime factor and its amount. Time complexity is O(logn).
    pub fn factorize(&self, mut n: usize) -> Vec<(usize, usize)> {
        assert!(n < self.spf.len());
        let mut map = BTreeMap::new();
        while self.spf[n] != n {
            *map.entry(self.spf[n]).or_insert(0) += 1;
            n /= self.spf[n];
        }
        *map.entry(self.spf[n]).or_insert(0) += 1;
        map.into_iter().collect()
    }
}

// Calculate x^n mod p. Time complexity is O(logn).
fn pow_mod(x: u128, mut n: u128, p: u128) -> u128 {
    let mut a = x % p;
    let mut res = 1;
    while n != 0 {
        if n & 1 != 0 {
            res = (res * a) % p;
        }
        a = (a * a) % p;
        n >>= 1;
    }
    res
}

/// Check if the given integer is prime or not based on Miller-Rabin primality test.
/// Time complexity is O(logn).
pub fn is_prime(n: u64) -> bool {
    // reference:
    // * https://miller-rabin.appspot.com
    // * https://drken1215.hatenablog.com/entry/2023/05/23/233000
    const A: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];

    match n {
        0 | 1 => return false,
        2 => return true,
        n if n & 1 == 0 => return false,
        _ => (),
    }

    let (s, d) = {
        let mut n = n - 1;
        let mut s = 0;
        while n & 1 == 0 {
            s += 1;
            n >>= 1;
        }
        (s, n)
    };
    for a in A.into_iter().map(|a| a % n).take_while(|a| a % n != 0) {
        let x = pow_mod(a as u128, d as u128, n as u128) as u64;
        if x != 1 {
            let xs = successors(Some(x), |&x| {
                Some(((x as u128 * x as u128) % n as u128) as u64)
            });
            if xs.take(s).all(|x| x != n - 1) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod test {
    use super::{is_prime, prime_factorize, PrimeFactorizer};

    #[test]
    fn test_prime_factorize() {
        assert_eq!(prime_factorize(5), vec![(5, 1)]);
        assert_eq!(prime_factorize(998244353), vec![(998244353, 1)]);
        assert_eq!(prime_factorize(120), vec![(2, 3), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_prime_factorizer() {
        let pf = PrimeFactorizer::new(1000);
        assert_eq!(pf.factorize(5), vec![(5, 1)]);
        assert_eq!(pf.factorize(120), vec![(2, 3), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(5));
        assert!(is_prime(998244353));
        assert!(!is_prime(u64::MAX));
        assert!(!is_prime(120));
    }
}
