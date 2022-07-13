/// Element-wise addition of two slices.
#[inline]
pub(crate) fn add_slice(lhs: &mut [f32], rhs: &[f32]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l += *r);
}

/// Element-wise subtraction of two slices.
#[inline]
pub(crate) fn sub_slice(lhs: &mut [f32], rhs: &[f32]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l -= *r);
}

/// Element-wise multiplication of two slices.
#[inline]
pub(crate) fn mul_slice(lhs: &mut [f32], rhs: &[f32]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l *= *r);
}

/// Element-wise division of two slices.
///
/// When the denominator is zero, the `default` value is assigned to the `lhs`.
#[inline]
pub(crate) fn div_slice(lhs: &mut [f32], rhs: &[f32], default: f32) {
    lhs.iter_mut()
        .zip(rhs)
        .for_each(|(l, r)| *l = if *r == 0.0 { default } else { *l / *r });
}

/// Multiply a scalar to a slice.
#[inline]
pub(crate) fn mul_slice_scalar(slice: &mut [f32], scalar: f32) {
    slice.iter_mut().for_each(|l| *l *= scalar);
}

/// Returns the "row" slice of a slice representing a two dimensional matrix.
#[inline]
pub(crate) fn row(slice: &[f32], index: usize, row_size: usize) -> &[f32] {
    &slice[index * row_size..(index + 1) * row_size]
}

/// Returns the mutable "row" slice of a slice representing a two dimensional matrix.
#[inline]
pub(crate) fn row_mut(slice: &mut [f32], index: usize, row_size: usize) -> &mut [f32] {
    &mut slice[index * row_size..(index + 1) * row_size]
}
