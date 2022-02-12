use num_traits::Zero;
use std::ops::{AddAssign, Div, DivAssign, MulAssign, SubAssign};

/// Element-wise addition of two slices.
///
/// # Examples
/// ```
/// use postflop_solver::add_slice;
///
/// let mut a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
///
/// add_slice(&mut a, &b);
///
/// assert_eq!(a, [5.0, 7.0, 9.0]);
/// ```
#[inline]
pub fn add_slice<T: Copy + AddAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l += *r);
}

/// Element-wise subtraction of two slices.
///
/// # Examples
/// ```
/// use postflop_solver::sub_slice;
///
/// let mut a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
///
/// sub_slice(&mut a, &b);
///
/// assert_eq!(a, [-3.0, -3.0, -3.0]);
/// ```
#[inline]
pub fn sub_slice<T: Copy + SubAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l -= *r);
}

/// Element-wise multiplication of two slices.
///
/// # Examples
/// ```
/// use postflop_solver::mul_slice;
///
/// let mut a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
///
/// mul_slice(&mut a, &b);
///
/// assert_eq!(a, [4.0, 10.0, 18.0]);
/// ```
#[inline]
pub fn mul_slice<T: Copy + MulAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l *= *r);
}

/// Element-wise division of two slices.
///
/// When the denominator is zero, the `default` value is assigned to the `lhs`.
///
/// # Examples
/// ```
/// use postflop_solver::div_slice;
///
/// let mut a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
///
/// div_slice(&mut a, &b, 0.0);
///
/// assert_eq!(a, [0.25, 0.4, 0.5]);
/// ```
#[inline]
pub fn div_slice<T: Copy + Div<Output = T> + Zero>(lhs: &mut [T], rhs: &[T], default: T) {
    lhs.iter_mut()
        .zip(rhs)
        .for_each(|(l, r)| *l = if r.is_zero() { default } else { *l / *r });
}

/// Add a scalar to a slice.
///
/// # Examples
/// ```
/// use postflop_solver::add_slice_scalar;
///
/// let mut a = [1.0, 2.0, 3.0];
///
/// add_slice_scalar(&mut a, 4.0);
///
/// assert_eq!(a, [5.0, 6.0, 7.0]);
/// ```
#[inline]
pub fn add_slice_scalar<T: Copy + AddAssign>(slice: &mut [T], scalar: T) {
    slice.iter_mut().for_each(|l| *l += scalar);
}

/// Subtract a scalar from a slice.
///
/// # Examples
/// ```
/// use postflop_solver::sub_slice_scalar;
///
/// let mut a = [1.0, 2.0, 3.0];
///
/// sub_slice_scalar(&mut a, 4.0);
///
/// assert_eq!(a, [-3.0, -2.0, -1.0]);
/// ```
#[inline]
pub fn sub_slice_scalar<T: Copy + SubAssign>(slice: &mut [T], scalar: T) {
    slice.iter_mut().for_each(|l| *l -= scalar);
}

/// Multiply a scalar to a slice.
///
/// # Examples
/// ```
/// use postflop_solver::mul_slice_scalar;
///
/// let mut a = [1.0, 2.0, 3.0];
///
/// mul_slice_scalar(&mut a, 4.0);
///
/// assert_eq!(a, [4.0, 8.0, 12.0]);
/// ```
#[inline]
pub fn mul_slice_scalar<T: Copy + MulAssign>(slice: &mut [T], scalar: T) {
    slice.iter_mut().for_each(|l| *l *= scalar);
}

/// Divide a slice by a scalar.
///
/// # Examples
/// ```
/// use postflop_solver::div_slice_scalar;
///
/// let mut a = [1.0, 2.0, 3.0];
///
/// div_slice_scalar(&mut a, 4.0);
///
/// assert_eq!(a, [0.25, 0.5, 0.75]);
/// ```
#[inline]
pub fn div_slice_scalar<T: Copy + DivAssign>(slice: &mut [T], scalar: T) {
    slice.iter_mut().for_each(|l| *l /= scalar);
}

/// Returns the "row" slice of a slice representing a two dimensional matrix.
///
/// # Examples
/// ```
/// use postflop_solver::row;
///
/// // An array representing a 2x2 matrix
/// let a = [1.0, 2.0, 3.0, 4.0];
///
/// // extract the second row
/// let r = row(&a, 1, 2);
///
/// assert_eq!(r, [3.0, 4.0]);
/// ```
#[inline]
pub fn row<T>(slice: &[T], index: usize, row_size: usize) -> &[T] {
    &slice[index * row_size..(index + 1) * row_size]
}

/// Returns the mutable "row" slice of a slice representing a two dimensional matrix.
///
/// # Examples
/// ```
/// use postflop_solver::row_mut;
///
/// // An array representing a 2x2 matrix
/// let mut a = [1.0, 2.0, 3.0, 4.0];
///
/// // extract the second row
/// let r = row_mut(&mut a, 1, 2);
///
/// // modify the second row
/// r[0] = 5.0;
///
/// assert_eq!(a, [1.0, 2.0, 5.0, 4.0]);
/// ```
#[inline]
pub fn row_mut<T>(slice: &mut [T], index: usize, row_size: usize) -> &mut [T] {
    &mut slice[index * row_size..(index + 1) * row_size]
}
