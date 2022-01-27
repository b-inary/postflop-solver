use num_traits::Zero;
use std::ops::{AddAssign, Div, DivAssign, MulAssign, SubAssign};

#[inline]
pub fn add_slice<T: Copy + AddAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l += *r);
}

#[inline]
pub fn sub_slice<T: Copy + SubAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l -= *r);
}

#[inline]
pub fn mul_slice<T: Copy + MulAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l *= *r);
}

#[inline]
pub fn div_slice<T: Copy + Div<Output = T> + Zero>(lhs: &mut [T], rhs: &[T], default: T) {
    lhs.iter_mut()
        .zip(rhs)
        .for_each(|(l, r)| *l = if r.is_zero() { default } else { *l / *r });
}

#[inline]
pub fn add_slice_scalar<T: Copy + AddAssign>(lhs: &mut [T], rhs: T) {
    lhs.iter_mut().for_each(|l| *l += rhs);
}

#[inline]
pub fn sub_slice_scalar<T: Copy + SubAssign>(lhs: &mut [T], rhs: T) {
    lhs.iter_mut().for_each(|l| *l -= rhs);
}

#[inline]
pub fn mul_slice_scalar<T: Copy + MulAssign>(lhs: &mut [T], rhs: T) {
    lhs.iter_mut().for_each(|l| *l *= rhs);
}

#[inline]
pub fn div_slice_scalar<T: Copy + DivAssign>(lhs: &mut [T], rhs: T) {
    lhs.iter_mut().for_each(|l| *l /= rhs);
}

#[inline]
pub fn row<T>(slice: &[T], index: usize, row_size: usize) -> &[T] {
    &slice[index * row_size..(index + 1) * row_size]
}

#[inline]
pub fn row_mut<T>(slice: &mut [T], index: usize, row_size: usize) -> &mut [T] {
    &mut slice[index * row_size..(index + 1) * row_size]
}
