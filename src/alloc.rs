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

pub static STACK_UNIT_SIZE: AtomicUsize = AtomicUsize::new(0);

pub struct StackAlloc;

struct StackAllocData {
    unit_capacity: usize,
    index: usize,
    base: Vec<usize>,
    current: Vec<usize>,
}

thread_local! {
    static STACK_ALLOC_DATA: RefCell<StackAllocData> = RefCell::new(StackAllocData {
        unit_capacity: 0,
        index: 0,
        base: Vec::new(),
        current: Vec::new(),
    });
}

impl StackAllocData {
    #[inline]
    fn init(&mut self) {
        let size = STACK_UNIT_SIZE.load(Ordering::Relaxed);
        if self.unit_capacity != size {
            self.free();
            let layout = Layout::from_size_align(size, ALIGNMENT).unwrap();
            let ptr = unsafe { alloc::alloc(layout) } as usize;
            self.unit_capacity = size;
            self.index = 0;
            self.base.push(ptr);
            self.current.push(ptr);
        }
    }

    #[inline]
    fn free(&mut self) {
        let layout = Layout::from_size_align(self.unit_capacity, ALIGNMENT).unwrap();
        for b in &self.base {
            unsafe { alloc::dealloc(*b as *mut u8, layout) };
        }
        self.unit_capacity = 0;
        self.index = 0;
        self.base.clear();
        self.current.clear();
    }

    #[inline]
    fn allocate(&mut self, size: usize) -> *mut [u8] {
        self.init();
        let size = align_up(size);
        if self.current[self.index] + size > self.base[self.index] + self.unit_capacity {
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
        self.index += 1;
        if self.index == self.base.len() {
            let layout = Layout::from_size_align(self.unit_capacity, ALIGNMENT).unwrap();
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
        STACK_ALLOC_DATA.with(|data| {
            let mut data = data.borrow_mut();
            if layout.align() <= ALIGNMENT {
                // perfoms allocation
                Ok(NonNull::new(data.allocate(layout.size())).unwrap())
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
            data.deallocate(ptr.as_ptr(), layout.size());
        })
    }
}
