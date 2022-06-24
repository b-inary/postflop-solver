use num_traits::Zero;
use std::ops::{AddAssign, Div, MulAssign, SubAssign};

/// Element-wise addition of two slices.
#[inline]
pub fn add_slice<T: Copy + AddAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l += *r);
}

/// Element-wise subtraction of two slices.
#[inline]
pub fn sub_slice<T: Copy + SubAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l -= *r);
}

/// Element-wise multiplication of two slices.
#[inline]
pub fn mul_slice<T: Copy + MulAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l *= *r);
}

/// Element-wise division of two slices.
///
/// When the denominator is zero, the `default` value is assigned to the `lhs`.
#[inline]
pub fn div_slice<T: Copy + Div<Output = T> + Zero>(lhs: &mut [T], rhs: &[T], default: T) {
    lhs.iter_mut()
        .zip(rhs)
        .for_each(|(l, r)| *l = if r.is_zero() { default } else { *l / *r });
}

/// Multiply a scalar to a slice.
#[inline]
pub fn mul_slice_scalar<T: Copy + MulAssign>(slice: &mut [T], scalar: T) {
    slice.iter_mut().for_each(|l| *l *= scalar);
}

/// Returns the "row" slice of a slice representing a two dimensional matrix.
#[inline]
pub fn row<T>(slice: &[T], index: usize, row_size: usize) -> &[T] {
    &slice[index * row_size..(index + 1) * row_size]
}

/// Returns the mutable "row" slice of a slice representing a two dimensional matrix.
#[inline]
pub fn row_mut<T>(slice: &mut [T], index: usize, row_size: usize) -> &mut [T] {
    &mut slice[index * row_size..(index + 1) * row_size]
}
