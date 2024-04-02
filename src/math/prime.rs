use std::{collections::BTreeMap, iter::successors};

/// Find all prime factor and its amount. Time complexity is O(√n)
pub fn prime_factorize(mut n: u64) -> Vec<(u64, usize)> {
    let mut res = Vec::new();
    let mut k = 2;
    while k * k <= n {
        if n % k == 0 {
            let mut count = 0;
            while n % k == 0 {
                n /= k;
                count += 1;
            }
            res.push((k, count));
        }
        k += 1;
    }
    if n != 1 {
        res.push((n, 1));
    }
    res
}

/// A struct to perform prime factorize by pre-calculated table.
#[derive(Clone)]
pub struct PrimeFactorizer {
    spf: Vec<usize>,
}

impl PrimeFactorizer {
    /// Prepare for multiple prime factorization query. Time complexity is O(nloglogn)
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

    /// Find all prime factor and its amount. Time complexity is O(logn)
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

// Calculate x^n mod p. Time complexity is O(logn)
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
/// Time complexity is O(logn)
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
