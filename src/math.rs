use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

macro_rules! impl_uint {
    ($($type:ident),* $(,)+) => {$(
        impl Integral for $type {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn abs(self) -> Self { self }
        }
    )*};
}

macro_rules! impl_int {
    ($($type:ident),* $(,)+) => {$(
        impl Integral for $type {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn abs(self) -> Self { self.abs() }
        }
    )*};
}

impl_uint! {
    u8, u16, u32, u64, u128, usize,
}

impl_int! {
    i8, i16, i32, i64, i128, isize,
}

// NOTE: This trait is subject to change. Some functions may be removed or added.
pub trait Integral:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + RemAssign<Self>
    + Eq
    + Ord
    + Copy
    + Sized
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
}

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
    use super::{gcd, is_prime, lcm, prime_factorize};

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
}
