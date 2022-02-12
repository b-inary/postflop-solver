use std::alloc::{self, AllocError, Allocator, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

const ALIGNMENT: usize = 16;

#[inline]
pub fn align_up(size: usize) -> usize {
    let mask = ALIGNMENT - 1;
    (size + mask) & !mask
}

pub static STACK_SIZE: AtomicUsize = AtomicUsize::new(0);

pub struct StackAlloc;

struct StackAllocData {
    base: usize,
    size: usize,
    current: usize,
}

thread_local! {
    static STACK_ALLOC_DATA: RefCell<StackAllocData> = RefCell::new(StackAllocData {
        base: 0,
        size: 0,
        current: 0,
    });
}

impl StackAllocData {
    #[inline]
    fn reallocate(&mut self, size: usize) {
        self.deallocate();
        let layout = Layout::from_size_align(size, ALIGNMENT).unwrap();
        self.base = unsafe { alloc::alloc(layout) } as usize;
        self.size = size;
        self.current = self.base;
    }

    #[inline]
    fn deallocate(&mut self) {
        if self.base != 0 {
            let layout = Layout::from_size_align(self.size, ALIGNMENT).unwrap();
            unsafe { alloc::dealloc(self.base as *mut u8, layout) };
        }
    }
}

impl Drop for StackAllocData {
    #[inline]
    fn drop(&mut self) {
        self.deallocate();
    }
}

unsafe impl Allocator for StackAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        STACK_ALLOC_DATA.with(|data| {
            let mut data = data.borrow_mut();

            // checks the stack size
            if data.size != STACK_SIZE.load(Ordering::Relaxed) {
                data.reallocate(STACK_SIZE.load(Ordering::Relaxed));
            }

            // perfoms the allocation
            if layout.align() <= ALIGNMENT {
                let ptr = data.current as *mut u8;
                let slice = unsafe { slice::from_raw_parts_mut(ptr, layout.size()) };
                data.current += align_up(layout.size());

                // checks if the allocation is within the stack
                if data.current - data.base > data.size {
                    return Err(AllocError);
                }

                Ok(NonNull::new(slice as *mut [u8]).unwrap())
            } else {
                // unsupported alignment
                Err(AllocError)
            }
        })
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        STACK_ALLOC_DATA.with(|data| {
            let mut data = data.borrow_mut();
            let ptr = ptr.as_ptr() as usize;

            // checks if the pointer indicates the last allocation
            if ptr != data.current - align_up(layout.size()) {
                panic!("deallocation error");
            }

            data.current = ptr;
        })
    }
}
