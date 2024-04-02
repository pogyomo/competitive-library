use super::Integral;

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
    use super::{gcd, lcm};

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
