use std::alloc::{self, AllocError, Allocator, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::slice;

const ALIGNMENT: usize = 16;
const STACK_UNIT: usize = 1 << 20; // 1MB

#[inline]
pub(crate) fn align_up(size: usize) -> usize {
    let mask = ALIGNMENT - 1;
    (size + mask) & !mask
}

/// A custom memory allocator that allocates memory in a stack-like manner.
///
/// This allocator can be used to reduce the number of calls of the default allocator.
/// In particular, the default allocator is not efficient in a multi-threaded WASM environment, so
/// using this allocator can significantly improve the performance in such an environment.
/// Note that this allocator assumes that `allocate` and `deallocate` are called in a stack-like
/// manner, and it panics if this assumption is not satisfied.
#[derive(Clone)]
pub(crate) struct StackAlloc;

struct StackAllocData {
    index: usize,
    base: Vec<usize>,
    current: Vec<usize>,
}

thread_local! {
    static STACK_ALLOC_DATA: RefCell<StackAllocData> = RefCell::new(StackAllocData {
        index: usize::MAX,
        base: Vec::new(),
        current: Vec::new(),
    });
}

impl StackAllocData {
    #[inline]
    fn free(&mut self) {
        if self.index == usize::MAX {
            return;
        }
        if self.index != 0 || self.base.first() != self.current.first() {
            panic!("freeing error");
        }
        let layout = Layout::from_size_align(STACK_UNIT, ALIGNMENT).unwrap();
        for b in &self.base {
            unsafe { alloc::dealloc(*b as *mut u8, layout) };
        }
        self.index = usize::MAX;
        self.base.clear();
        self.current.clear();
        self.base.shrink_to_fit();
        self.current.shrink_to_fit();
    }

    #[inline]
    fn allocate(&mut self, size: usize) -> *mut [u8] {
        let size = align_up(size);
        let index = self.index;
        if index == usize::MAX || self.current[index] + size > self.base[index] + STACK_UNIT {
            self.increment_index();
        }
        let ptr = self.current[self.index] as *mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, size) };
        self.current[self.index] += size;
        slice as *mut [u8]
    }

    #[inline]
    fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        let ptr = ptr as usize;
        self.current[self.index] -= align_up(size);
        if self.current[self.index] != ptr {
            panic!("deallocation error");
        }
        if self.base[self.index] == ptr {
            self.decrement_index();
        }
    }

    #[inline]
    fn increment_index(&mut self) {
        self.index = (self.index as isize + 1) as usize;
        if self.index == self.base.len() {
            let layout = Layout::from_size_align(STACK_UNIT, ALIGNMENT).unwrap();
            let ptr = unsafe { alloc::alloc(layout) } as usize;
            self.base.push(ptr);
            self.current.push(ptr);
        }
    }

    #[inline]
    fn decrement_index(&mut self) {
        if self.index > 0 {
            self.index -= 1;
        }
    }
}

impl Drop for StackAllocData {
    #[inline]
    fn drop(&mut self) {
        self.free();
    }
}

unsafe impl Allocator for StackAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() > STACK_UNIT || layout.align() > ALIGNMENT {
            return Err(AllocError);
        }

        STACK_ALLOC_DATA.with(|data| {
            let mut data = data.borrow_mut();
            Ok(NonNull::new(data.allocate(layout.size())).unwrap())
        })
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        STACK_ALLOC_DATA.with(|data| {
            let mut data = data.borrow_mut();
            data.deallocate(ptr.as_ptr(), layout.size());
        })
    }
}

pub(crate) fn free_custom_alloc_buffer() {
    STACK_ALLOC_DATA.with(|data| {
        let mut data = data.borrow_mut();
        data.free();
    });
}
