pub mod theory;

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
