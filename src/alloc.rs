use std::alloc::{self, AllocError, Allocator, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const MAX_ALIGNMENT: usize = 16;

pub struct StackAlloc;

pub struct StackAllocData {
    base: usize,
    size: usize,
    current: usize,
}

pub static STACK_SIZE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static STACK_ALLOC_DATA: RefCell<StackAllocData> = RefCell::new(StackAllocData {
        base: 0,
        size: 0,
        current: 0,
    });
}

pub fn align_up(size: usize) -> usize {
    let mask = MAX_ALIGNMENT - 1;
    (size + mask) & !mask
}

impl StackAllocData {
    fn reallocate(&mut self, size: usize) {
        let layout = Layout::from_size_align(size, MAX_ALIGNMENT).unwrap();
        self.base = unsafe { alloc::alloc(layout) } as usize;
        self.size = size;
        self.current = self.base;
    }
}

unsafe impl Allocator for StackAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        STACK_ALLOC_DATA.with(|data| {
            let mut data = data.borrow_mut();

            // checks the stack size
            if data.size != STACK_SIZE.load(Ordering::Relaxed) {
                data.reallocate(STACK_SIZE.load(Ordering::Relaxed));
            }

            // perfoms the allocation
            if layout.align() <= MAX_ALIGNMENT {
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
