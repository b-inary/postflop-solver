use super::*;
use std::slice;

/// A struct representing a node in a postflop game tree.
///
/// The nodes must be stored as `Vec<MutexLike<PostFlopNode>>`.
#[derive(Debug, Clone, Copy)]
pub struct PostFlopNode {
    pub(super) prev_action: Action,
    pub(super) player: u8,
    pub(super) turn: u8,
    pub(super) river: u8,
    pub(super) is_locked: bool,
    pub(super) amount: i32,
    pub(super) children_offset: u32,
    pub(super) num_children: u32,
    pub(super) storage1: *mut u8,
    pub(super) storage2: *mut u8,
    pub(super) num_elements: u32,
    pub(super) num_elements_aux: u32,
    pub(super) scale1: f32,
    pub(super) scale2: f32,
}

unsafe impl Send for PostFlopNode {}
unsafe impl Sync for PostFlopNode {}

impl GameNode for PostFlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player & PLAYER_CHANCE_FLAG != 0
    }

    #[inline]
    fn cfvalue_storage(&self, player: usize) -> CfValueStorage {
        let last_player = self.player & (PLAYER_MASK | PLAYER_CHANCE);
        let allin_flag = self.player & PLAYER_ALLIN_FLAG != 0;
        let pair = match (allin_flag, last_player) {
            (false, PLAYER_OOP) => [CfValueStorage::None, CfValueStorage::All],
            (false, PLAYER_IP) => [CfValueStorage::Sum, CfValueStorage::All],
            (true, PLAYER_OOP) => [CfValueStorage::None, CfValueStorage::Sum],
            (true, PLAYER_IP) => [CfValueStorage::Sum, CfValueStorage::None],
            _ => [CfValueStorage::None, CfValueStorage::None],
        };
        pair[player]
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    #[inline]
    fn chance_factor(&self) -> f32 {
        match self.turn {
            NOT_DEALT => 1.0 / 45.0,
            _ => 1.0 / 44.0,
        }
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children()[action].lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance(&self, player: usize) -> &[f32] {
        let (base, len) = match player {
            0 => (self.storage1, self.num_elements),
            _ => (self.storage2, self.num_elements_aux),
        };
        unsafe { slice::from_raw_parts(base as *const f32, len as usize) }
    }

    #[inline]
    fn cfvalues_chance_mut(&mut self, player: usize) -> &mut [f32] {
        let (base, len) = match player {
            0 => (self.storage1, self.num_elements),
            _ => (self.storage2, self.num_elements_aux),
        };
        unsafe { slice::from_raw_parts_mut(base as *mut f32, len as usize) }
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u16, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_compressed(&self, player: usize) -> &[i16] {
        let (base, len) = match player {
            0 => (self.storage1, self.num_elements),
            _ => (self.storage2, self.num_elements_aux),
        };
        unsafe { slice::from_raw_parts(base as *const i16, len as usize) }
    }

    #[inline]
    fn cfvalues_chance_compressed_mut(&mut self, player: usize) -> &mut [i16] {
        let (base, len) = match player {
            0 => (self.storage1, self.num_elements),
            _ => (self.storage2, self.num_elements_aux),
        };
        unsafe { slice::from_raw_parts_mut(base as *mut i16, len as usize) }
    }

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn regret_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_regret_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_cfvalue_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[doc(hidden)]
    fn cfvalue_chance_scale(&self, player: usize) -> f32 {
        match player {
            0 => self.scale1,
            _ => self.scale2,
        }
    }

    #[doc(hidden)]
    fn set_cfvalue_chance_scale(&mut self, player: usize, scale: f32) {
        match player {
            0 => self.scale1 = scale,
            _ => self.scale2 = scale,
        }
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }
}

impl Default for PostFlopNode {
    #[inline]
    fn default() -> Self {
        Self {
            prev_action: Action::None,
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            is_locked: false,
            amount: 0,
            children_offset: 0,
            num_children: 0,
            storage1: ptr::null_mut(),
            storage2: ptr::null_mut(),
            num_elements: 0,
            num_elements_aux: 0,
            scale1: 0.0,
            scale2: 0.0,
        }
    }
}

impl PostFlopNode {
    pub(super) fn children(&self) -> &[MutexLike<PostFlopNode>] {
        // This is safe because `MutexLike<T>` is a `repr(transparent)` wrapper around `T`.
        let self_ptr = self as *const _ as *const MutexLike<PostFlopNode>;
        unsafe {
            slice::from_raw_parts(
                self_ptr.add(self.children_offset as usize),
                self.num_children as usize,
            )
        }
    }
}
