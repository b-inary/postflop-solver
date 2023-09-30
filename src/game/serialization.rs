use super::*;

use crate::interface::*;
use crate::utility::*;
use std::cell::Cell;
use std::ptr;

use bincode::{
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
};

impl PostFlopGame {
    /// Returns the storage mode of this instance.
    ///
    /// The storage mode represents the deepest accessible node in the game tree.
    /// For example, if the storage mode is `BoardState::Turn`, then the game tree
    /// contains no information after the river deal.
    #[inline]
    pub fn storage_mode(&self) -> BoardState {
        self.storage_mode
    }

    /// Returns the target storage mode, which is used for serialization.
    #[inline]
    pub fn target_storage_mode(&self) -> BoardState {
        self.target_storage_mode
    }

    /// Sets the target storage mode.
    #[inline]
    pub fn set_target_storage_mode(&mut self, mode: BoardState) -> Result<(), String> {
        if mode > self.storage_mode {
            return Err("Cannot set target to a higher value than the current storage".to_string());
        }

        if mode < self.tree_config.initial_state {
            return Err("Cannot set target to a lower value than the initial state".to_string());
        }

        self.target_storage_mode = mode;
        Ok(())
    }

    /// Returns the memory usage when the target storage mode is used for serialization.
    #[inline]
    pub fn target_memory_usage(&self) -> u64 {
        match self.target_storage_mode {
            BoardState::River => match self.is_compression_enabled {
                false => self.memory_usage().0,
                true => self.memory_usage().1,
            },
            _ => {
                let num_target_storage = self.num_target_storage();
                num_target_storage.iter().map(|&x| x as u64).sum::<u64>() + self.misc_memory_usage
            }
        }
    }

    /// Returns the number of storage elements required for the target storage mode.
    fn num_target_storage(&self) -> [usize; 4] {
        if self.state <= State::TreeBuilt {
            return [0; 4];
        }

        let num_bytes = if self.is_compression_enabled { 2 } else { 4 };
        if self.target_storage_mode == BoardState::River {
            // omit storing the counterfactual values
            return [num_bytes * self.num_storage as usize, 0, 0, 0];
        }

        let mut node_index = match self.target_storage_mode {
            BoardState::Flop => self.num_nodes[0],
            _ => self.num_nodes[0] + self.num_nodes[1],
        } as usize;

        let mut num_storage = [0; 4];

        while num_storage.iter().any(|&x| x == 0) {
            node_index -= 1;
            let node = self.node_arena[node_index].lock();
            if num_storage[0] == 0 && !node.is_terminal() && !node.is_chance() {
                let offset = unsafe { node.storage1.offset_from(self.storage1.as_ptr()) };
                let offset_ip = unsafe { node.storage3.offset_from(self.storage_ip.as_ptr()) };
                let len = num_bytes * node.num_elements as usize;
                let len_ip = num_bytes * node.num_elements_ip as usize;
                num_storage[0] = offset as usize + len;
                num_storage[1] = offset as usize + len;
                num_storage[2] = offset_ip as usize + len_ip;
            }
            if num_storage[3] == 0 && node.is_chance() {
                let offset = unsafe { node.storage1.offset_from(self.storage_chance.as_ptr()) };
                let len = num_bytes * node.num_elements as usize;
                num_storage[3] = offset as usize + len;
            }
        }

        num_storage
    }
}

static VERSION_STR: &str = "2023-03-19";

thread_local! {
    static PTR_BASE: Cell<[*const u8; 2]> = Cell::new([ptr::null(); 2]);
    static CHANCE_BASE: Cell<*const u8> = Cell::new(ptr::null());
    static PTR_BASE_MUT: Cell<[*mut u8; 3]> = Cell::new([ptr::null_mut(); 3]);
    static CHANCE_BASE_MUT: Cell<*mut u8> = Cell::new(ptr::null_mut());
}

impl Encode for PostFlopGame {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        if self.state <= State::Uninitialized {
            return Err(EncodeError::Other("Game is not successfully initialized"));
        }

        let num_storage = self.num_target_storage();

        // version
        VERSION_STR.to_string().encode(encoder)?;

        // contents
        self.state.encode(encoder)?;
        self.card_config.encode(encoder)?;
        self.tree_config.encode(encoder)?;
        self.added_lines.encode(encoder)?;
        self.removed_lines.encode(encoder)?;
        self.action_root.encode(encoder)?;
        self.target_storage_mode.encode(encoder)?;
        self.num_nodes.encode(encoder)?;
        self.is_compression_enabled.encode(encoder)?;
        self.num_storage.encode(encoder)?;
        self.num_storage_ip.encode(encoder)?;
        self.num_storage_chance.encode(encoder)?;
        self.misc_memory_usage.encode(encoder)?;
        self.storage1[0..num_storage[0]].encode(encoder)?;
        self.storage2[0..num_storage[1]].encode(encoder)?;
        self.storage_ip[0..num_storage[2]].encode(encoder)?;
        self.storage_chance[0..num_storage[3]].encode(encoder)?;

        let num_nodes = match self.target_storage_mode {
            BoardState::Flop => self.num_nodes[0] as usize,
            BoardState::Turn => (self.num_nodes[0] + self.num_nodes[1]) as usize,
            BoardState::River => self.node_arena.len(),
        };

        // locking strategy (need to filter)
        let mut locking_strategy = self.locking_strategy.clone();
        locking_strategy.retain(|&i, _| i < num_nodes);
        locking_strategy.encode(encoder)?;

        // store base pointers
        PTR_BASE.with(|c| {
            if self.state >= State::MemoryAllocated {
                c.set([self.storage1.as_ptr(), self.storage_ip.as_ptr()]);
            } else {
                c.set([ptr::null(); 2]);
            }
        });

        CHANCE_BASE.with(|c| {
            if self.state >= State::MemoryAllocated {
                c.set(self.storage_chance.as_ptr());
            } else {
                c.set(ptr::null());
            }
        });

        // game tree
        self.node_arena[0..num_nodes].encode(encoder)?;

        Ok(())
    }
}

impl Decode for PostFlopGame {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        // version check
        let version = String::decode(decoder)?;
        if version != VERSION_STR {
            return Err(DecodeError::OtherString(format!(
                "Version mismatch: expected '{VERSION_STR}', but got '{version}'"
            )));
        }

        // game instance
        let mut game = Self {
            state: Decode::decode(decoder)?,
            card_config: Decode::decode(decoder)?,
            tree_config: Decode::decode(decoder)?,
            added_lines: Decode::decode(decoder)?,
            removed_lines: Decode::decode(decoder)?,
            action_root: Decode::decode(decoder)?,
            storage_mode: Decode::decode(decoder)?,
            num_nodes: Decode::decode(decoder)?,
            is_compression_enabled: Decode::decode(decoder)?,
            num_storage: Decode::decode(decoder)?,
            num_storage_ip: Decode::decode(decoder)?,
            num_storage_chance: Decode::decode(decoder)?,
            misc_memory_usage: Decode::decode(decoder)?,
            storage1: Decode::decode(decoder)?,
            storage2: Decode::decode(decoder)?,
            storage_ip: Decode::decode(decoder)?,
            storage_chance: Decode::decode(decoder)?,
            locking_strategy: Decode::decode(decoder)?,
            ..Default::default()
        };

        game.target_storage_mode = game.storage_mode;
        if game.storage_mode == BoardState::River && game.state >= State::MemoryAllocated {
            let num_bytes = if game.is_compression_enabled { 2 } else { 4 };
            game.storage2 = vec![0; (num_bytes * game.num_storage) as usize];
            game.storage_ip = vec![0; (num_bytes * game.num_storage_ip) as usize];
            game.storage_chance = vec![0; (num_bytes * game.num_storage_chance) as usize];
        }

        // store base pointers
        PTR_BASE_MUT.with(|c| {
            if game.state >= State::MemoryAllocated {
                c.set([
                    game.storage1.as_mut_ptr(),
                    game.storage2.as_mut_ptr(),
                    game.storage_ip.as_mut_ptr(),
                ]);
            } else {
                c.set([ptr::null_mut(); 3]);
            }
        });

        CHANCE_BASE_MUT.with(|c| {
            if game.state >= State::MemoryAllocated {
                c.set(game.storage_chance.as_mut_ptr());
            } else {
                c.set(ptr::null_mut());
            }
        });

        // game tree
        game.node_arena = Decode::decode(decoder)?;

        // initialization
        game.check_card_config().map_err(DecodeError::OtherString)?;
        game.init_card_fields();
        game.init_interpreter();
        game.back_to_root();

        // restore the counterfactual values
        if game.storage_mode == BoardState::River && game.state == State::Solved {
            game.state = State::MemoryAllocated;
            finalize(&mut game);
        }

        Ok(game)
    }
}

impl Encode for PostFlopNode {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // contents
        self.prev_action.encode(encoder)?;
        self.player.encode(encoder)?;
        self.turn.encode(encoder)?;
        self.river.encode(encoder)?;
        self.is_locked.encode(encoder)?;
        self.amount.encode(encoder)?;
        self.children_offset.encode(encoder)?;
        self.num_children.encode(encoder)?;
        self.num_elements_ip.encode(encoder)?;
        self.num_elements.encode(encoder)?;
        self.scale1.encode(encoder)?;
        self.scale2.encode(encoder)?;
        self.scale3.encode(encoder)?;

        // pointer offset
        if !self.storage1.is_null() {
            if self.is_terminal() {
                // do nothing
            } else if self.is_chance() {
                let base = CHANCE_BASE.with(|c| c.get());
                unsafe { self.storage1.offset_from(base).encode(encoder)? };
            } else {
                let bases = PTR_BASE.with(|c| c.get());
                unsafe {
                    self.storage1.offset_from(bases[0]).encode(encoder)?;
                    self.storage3.offset_from(bases[1]).encode(encoder)?;
                }
            }
        }

        Ok(())
    }
}

impl Decode for PostFlopNode {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
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
            num_elements_ip: Decode::decode(decoder)?,
            num_elements: Decode::decode(decoder)?,
            scale1: Decode::decode(decoder)?,
            scale2: Decode::decode(decoder)?,
            scale3: Decode::decode(decoder)?,
            ..Default::default()
        };

        // pointers
        if node.is_terminal() {
            // do nothing
        } else if node.is_chance() {
            let base = CHANCE_BASE_MUT.with(|c| c.get());
            if !base.is_null() {
                node.storage1 = unsafe { base.offset(isize::decode(decoder)?) };
            }
        } else {
            let bases = PTR_BASE_MUT.with(|c| c.get());
            if !bases[0].is_null() {
                let offset = isize::decode(decoder)?;
                let offset_ip = isize::decode(decoder)?;
                node.storage1 = unsafe { bases[0].offset(offset) };
                node.storage2 = unsafe { bases[1].offset(offset) };
                node.storage3 = unsafe { bases[2].offset(offset_ip) };
            }
        }

        Ok(node)
    }
}
