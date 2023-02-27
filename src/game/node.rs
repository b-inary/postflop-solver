use super::*;
use std::slice;

#[cfg(feature = "bincode")]
use std::cell::Cell;

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

#[cfg(feature = "bincode")]
thread_local! {
    pub(super) static ACTION_BASE: Cell<(*mut u8, *mut u8)> =
        Cell::new((ptr::null_mut(), ptr::null_mut()));
    pub(super) static CHANCE_BASE: Cell<*mut u8> = Cell::new(ptr::null_mut());
}

#[cfg(feature = "bincode")]
impl Encode for PostFlopNode {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // compute pointer offset
        let (offset, offset_aux) = if self.is_chance() {
            CHANCE_BASE.with(|p| {
                let base = p.get();
                let ret = match self.storage1.is_null() {
                    true => -1,
                    false => unsafe { self.storage1.offset_from(base) },
                };
                let ret_aux = match self.storage2.is_null() {
                    true => -1,
                    false => unsafe { self.storage2.offset_from(base) },
                };
                (ret, ret_aux)
            })
        } else {
            ACTION_BASE.with(|ps| {
                let (base1, _) = ps.get();
                match self.storage1.is_null() {
                    true => (-1, -1),
                    false => {
                        let ret = unsafe { self.storage1.offset_from(base1) };
                        (ret, -1)
                    }
                }
            })
        };

        // contents
        self.prev_action.encode(encoder)?;
        self.player.encode(encoder)?;
        self.turn.encode(encoder)?;
        self.river.encode(encoder)?;
        self.is_locked.encode(encoder)?;
        self.amount.encode(encoder)?;
        self.children_offset.encode(encoder)?;
        self.num_children.encode(encoder)?;
        self.num_elements.encode(encoder)?;
        self.num_elements_aux.encode(encoder)?;
        self.scale1.encode(encoder)?;
        self.scale2.encode(encoder)?;
        offset.encode(encoder)?;
        offset_aux.encode(encoder)?;

        Ok(())
    }
}

#[cfg(feature = "bincode")]
impl Decode for PostFlopNode {
    fn decode<D: bincode::de::Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        // node instance
        let mut node = Self {
            prev_action: Decode::decode(decoder)?,
            player: Decode::decode(decoder)?,
            turn: Decode::decode(decoder)?,
            river: Decode::decode(decoder)?,
            is_locked: Decode::decode(decoder)?,
            amount: Decode::decode(decoder)?,
            children_offset: Decode::decode(decoder)?,
            num_children: Decode::decode(decoder)?,
            num_elements: Decode::decode(decoder)?,
            num_elements_aux: Decode::decode(decoder)?,
            scale1: Decode::decode(decoder)?,
            scale2: Decode::decode(decoder)?,
            ..Default::default()
        };

        // pointers
        let offset = isize::decode(decoder)?;
        let offset_aux = isize::decode(decoder)?;
        if node.is_chance() {
            let base = CHANCE_BASE.with(|p| p.get());
            if offset >= 0 {
                node.storage1 = unsafe { base.offset(offset) };
            }
            if offset_aux >= 0 {
                node.storage2 = unsafe { base.offset(offset_aux) };
            }
        } else if offset >= 0 {
            let (base1, base2) = ACTION_BASE.with(|ps| ps.get());
            node.storage1 = unsafe { base1.offset(offset) };
            node.storage2 = unsafe { base2.offset(offset) };
        }

        Ok(node)
    }
}
