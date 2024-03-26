pub mod fft;
pub mod theory;

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

macro_rules! impl_uint {
    ($($type:ident),*) => {$(
        impl NumOps for $type {}
        impl Integral for $type {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn abs(self) -> Self { self }
        }
    )*};
}

macro_rules! impl_int {
    ($($type:ident),*) => {$(
        impl NumOps for $type {}
        impl Integral for $type {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn abs(self) -> Self { self.abs() }
        }
    )*};
}

macro_rules! impl_float {
    ($($type:ident),*) => {$(
        impl NumOps for $type {}
        impl Float for $type {
            const PI: $type = ::std::$type::consts::PI;

            fn zero() -> Self { 0.0 }
            fn one() -> Self { 1.0 }
            fn cos(self) -> Self { self.cos() }
            fn sin(self) -> Self { self.sin() }
            // TODO: I just casted usize to f32/f64. Should I return None if the cast lose infomation?
            fn from_usize(value: usize) -> Self { value as $type }
        }
    )*};
}

impl_uint! {
    u8, u16, u32, u64, u128, usize
}

impl_int! {
    i8, i16, i32, i64, i128, isize
}

impl_float! {
    f32, f64
}

// NOTE: These trait is subject to change. Some functions may be removed or added.

/// A collection of math operations.
pub trait NumOps:
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
    + Sized
{
}

/// A trait for integral.
pub trait Integral: NumOps + Eq + Ord + Copy + Sized {
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
}

/// A trait for floating point number.
pub trait Float: NumOps + Neg<Output = Self> + PartialEq + PartialOrd + Copy + Sized {
    const PI: Self;

    fn zero() -> Self;
    fn one() -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn from_usize(value: usize) -> Self; // TODO: Create a trait for conversion?
}

/// A struct represent complex number
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Complex<T> {
    re: T,
    im: T,
}

impl<T> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Float> Complex<T> {
    pub fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl<T: Float> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        (self * rhs.conjugate()) / (rhs.conjugate() * rhs.conjugate()).re
    }
}
