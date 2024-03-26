use super::{Complex, Float};
use std::iter::zip;

fn fft_impl<T: Float>(a: Vec<Complex<T>>, inv: T) -> Vec<Complex<T>> {
    assert!(a.len().is_power_of_two());
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
    let f = fft_impl(f, inv);
    let g = fft_impl(g, inv);

    let theta = (T::one() + T::one()) * T::PI * inv / T::from_usize(n);
    let zeta = Complex::new(T::cos(theta), T::sin(theta));

    let mut p = Complex::new(T::one(), T::zero());
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        res.push(f[i % (n >> 1)] + p * g[i % (n >> 1)]);
        p = p * zeta;
    }
    res
}

/// Perform fast fourier transform for `a`. We expect the size of `a` is power of two.
/// Time complexity is O(NlogN).
pub fn fft<T: Float>(a: Vec<Complex<T>>) -> Vec<Complex<T>> {
    fft_impl(a, T::one())
}

/// Perform inverse fast fourier transform for `a`. We expect the size of `a` is power of two.
/// Time complexity is O(NlogN).
pub fn ifft<T: Float>(a: Vec<Complex<T>>) -> Vec<Complex<T>> {
    let n = a.len();
    fft_impl(a, -T::one())
        .into_iter()
        .map(|a| a / T::from_usize(n))
        .collect()
}

/// Calculate convolution of given two sequences.
/// Time complexity is O(NlogN).
pub fn convolution<T: Float>(mut a: Vec<T>, mut b: Vec<T>) -> Vec<T> {
    let res_n = a.len() + b.len() - 1;
    let n = res_n.next_power_of_two();
    a.resize(n, T::zero());
    b.resize(n, T::zero());
    let da = fft(a.into_iter().map(|a| Complex::new(a, T::zero())).collect());
    let db = fft(b.into_iter().map(|b| Complex::new(b, T::zero())).collect());
    let dc = zip(da, db).map(|(a, b)| a * b).collect();
    ifft(dc).into_iter().take(res_n).map(|c| c.re).collect()
}

#[cfg(test)]
mod test {
    use super::convolution;

    #[test]
    fn test_convolution() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let ab = convolution(a, b)
            .into_iter()
            .map(|v: f64| v.round() as usize)
            .map(|v| v.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            ab,
            vec![
                String::from("2"),
                String::from("7"),
                String::from("16"),
                String::from("17"),
                String::from("12")
            ]
        );
    }
}
