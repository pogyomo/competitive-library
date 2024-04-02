use super::Integral;
use std::iter::successors;

/// Find all prime factor and its amount. Time complexity is O(√N).
/// For any `n`, prime_factorize(n) == prime_factorize(|n|).
pub fn prime_factorize<T: Integral>(n: T) -> Vec<(T, usize)> {
    let mut n = n.abs();
    let mut res = Vec::new();
    let mut k = T::one() + T::one();
    while k * k <= n {
        if n % k == T::zero() {
            let mut count = 0;
            while n % k == T::zero() {
                count += 1;
                n /= k;
            }
            res.push((k, count));
        }
        k += T::one();
    }
    if n != T::one() {
        res.push((n, 1));
    }
    res
}

/// Check if the given integer is prime or not. Time complexity is O(√N).
/// For any `n`, is_prime(n) == is_prime(|n|).
pub fn is_prime<T: Integral>(n: T) -> bool {
    let n = n.abs();
    let mut k = T::one() + T::one();
    while k * k <= n {
        if n % k == T::zero() {
            return false;
        }
        k += T::one();
    }
    true
}

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
pub fn fast_is_prime(n: u64) -> bool {
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

/// Find greatest common divider of `a` and `b`.
/// For any `a` and `b`, gcd(a, b) == gcd(|a|, |b|).
pub fn gcd<T: Integral>(a: T, b: T) -> T {
    let (a, b) = (a.abs(), b.abs());
    if b == T::zero() {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Find least common multiple of `a` and `b`.
/// For any `a` and `b`, lcm(a, b) == lcm(|a|, |b|).
pub fn lcm<T: Integral>(a: T, b: T) -> T {
    let (a, b) = (a.abs(), b.abs());
    (a / gcd(a, b)) * b
}

#[cfg(test)]
mod test {
    use super::{fast_is_prime, gcd, is_prime, lcm, prime_factorize};

    #[test]
    fn test_prime_factorize() {
        assert_eq!(prime_factorize(5), vec![(5, 1)]);
        assert_eq!(prime_factorize(998244353), vec![(998244353, 1)]);
        assert_eq!(prime_factorize(120), vec![(2, 3), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(5));
        assert!(is_prime(998244353));
        assert!(!is_prime(120));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(2, 5), 1);
        assert_eq!(gcd(10, 15), 5);
        assert_eq!(gcd(-10, -15), 5);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(2, 5), 10);
        assert_eq!(lcm(10, 15), 30);
        assert_eq!(lcm(-10, -15), 30);
    }

    #[test]
    fn test_fast_is_prime() {
        assert!(fast_is_prime(2));
        assert!(fast_is_prime(998244353));
        assert!(!fast_is_prime(u64::MAX));
        assert!(!fast_is_prime(120));
    }
}
