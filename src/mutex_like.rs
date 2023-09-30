use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "bincode")]
use bincode::{
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};

/// Mutex-like wrapper, but it actually does not perform any locking.
///
/// Use this wrapper when:
///   1. [`Send`], [`Sync`] and the interior mutability is needed,
///   2. it is (manually) guaranteed that data races will not occur, and
///   3. the performance is critical.
///
/// **Note**: This wrapper completely bypasses the "shared XOR mutable" rule of Rust.
/// Therefore, using this wrapper is **extremely unsafe** and should be avoided whenever possible.
#[derive(Debug)]
#[repr(transparent)]
pub struct MutexLike<T: ?Sized> {
    data: UnsafeCell<T>,
}

/// Smart pointer like wrapper that is returned when [`MutexLike`] is "locked".
#[derive(Debug)]
pub struct MutexGuardLike<'a, T: ?Sized + 'a> {
    mutex: &'a MutexLike<T>,
}

unsafe impl<T: ?Sized + Send> Send for MutexLike<T> {}
unsafe impl<T: ?Sized + Send> Sync for MutexLike<T> {}
unsafe impl<'a, T: ?Sized + Sync + 'a> Sync for MutexGuardLike<'a, T> {}

impl<T> MutexLike<T> {
    /// Creates a new [`MutexLike`] with the given value.
    ///
    /// # Examples
    /// ```
    /// use postflop_solver::MutexLike;
    ///
    /// let mutex_like = MutexLike::new(0);
    /// ```
    #[inline]
    pub fn new(val: T) -> Self {
        Self {
            data: UnsafeCell::new(val),
        }
    }
}

impl<T: ?Sized> MutexLike<T> {
    /// Acquires a mutex-like object **without** performing any locking.
    ///
    /// # Examples
    /// ```
    /// use postflop_solver::MutexLike;
    ///
    /// let mutex_like = MutexLike::new(0);
    /// *mutex_like.lock() = 10;
    /// assert_eq!(*mutex_like.lock(), 10);
    /// ```
    #[inline]
    pub fn lock(&self) -> MutexGuardLike<T> {
        MutexGuardLike { mutex: self }
    }
}

impl<T: ?Sized + Default> Default for MutexLike<T> {
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<'a, T: ?Sized + 'a> Deref for MutexGuardLike<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for MutexGuardLike<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

#[cfg(feature = "bincode")]
impl<T: Encode> Encode for MutexLike<T> {
    #[inline]
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.lock().encode(encoder)
    }
}

#[cfg(feature = "bincode")]
impl<T: Decode> Decode for MutexLike<T> {
    #[inline]
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self::new(T::decode(decoder)?))
    }
}

#[cfg(feature = "bincode")]
impl<'de, T: BorrowDecode<'de>> BorrowDecode<'de> for MutexLike<T> {
    #[inline]
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self::new(T::borrow_decode(decoder)?))
    }
}
