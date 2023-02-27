use super::*;

use bincode::error::{DecodeError, EncodeError};
use std::cell::Cell;

static VERSION_STR: &str = "2023-02-24";

thread_local! {
    static ACTION_BASE: Cell<(*mut u8, *mut u8)> = Cell::new((ptr::null_mut(), ptr::null_mut()));
    static CHANCE_BASE: Cell<*mut u8> = Cell::new(ptr::null_mut());
}

impl Encode for PostFlopGame {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // store base pointers
        ACTION_BASE.with(|ps| {
            if self.state >= State::MemoryAllocated {
                let base1 = match self.is_compression_enabled {
                    true => self.storage1_compressed.lock().as_mut_ptr() as *mut u8,
                    false => self.storage1.lock().as_mut_ptr() as *mut u8,
                };
                ps.set((base1, ptr::null_mut()));
            } else {
                ps.set((ptr::null_mut(), ptr::null_mut()));
            }
        });

        CHANCE_BASE.with(|p| {
            if self.state >= State::MemoryAllocated {
                let base = match self.is_compression_enabled {
                    true => self.storage_chance_compressed.lock().as_mut_ptr() as *mut u8,
                    false => self.storage_chance.lock().as_mut_ptr() as *mut u8,
                };
                p.set(base);
            } else {
                p.set(ptr::null_mut());
            }
        });

        // version
        VERSION_STR.to_string().encode(encoder)?;

        // action tree
        self.tree_config.encode(encoder)?;
        self.added_lines.encode(encoder)?;
        self.removed_lines.encode(encoder)?;

        // contents
        self.state.encode(encoder)?;
        self.card_config.encode(encoder)?;
        self.num_combinations.encode(encoder)?;
        self.is_compression_enabled.encode(encoder)?;
        self.misc_memory_usage.encode(encoder)?;
        self.num_storage_actions.encode(encoder)?;
        self.num_storage_chances.encode(encoder)?;
        self.storage1.encode(encoder)?;
        self.storage2.encode(encoder)?;
        self.storage_chance.encode(encoder)?;
        self.storage1_compressed.encode(encoder)?;
        self.storage2_compressed.encode(encoder)?;
        self.storage_chance_compressed.encode(encoder)?;
        self.locking_strategy.encode(encoder)?;
        self.root_cfvalue_ip.encode(encoder)?;
        self.history.encode(encoder)?;
        self.is_normalized_weight_cached.encode(encoder)?;

        // game tree
        self.node_arena.encode(encoder)?;

        Ok(())
    }
}

impl Decode for PostFlopGame {
    fn decode<D: bincode::de::Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        // version check
        let version = String::decode(decoder)?;
        if version != VERSION_STR {
            return Err(DecodeError::OtherString(format!(
                "Version mismatch: expected '{VERSION_STR}', but got '{version}'"
            )));
        }

        let tree_config = TreeConfig::decode(decoder)?;
        let added_lines = Vec::<Vec<Action>>::decode(decoder)?;
        let removed_lines = Vec::<Vec<Action>>::decode(decoder)?;

        let mut action_tree = ActionTree::new(tree_config).unwrap();
        for line in &added_lines {
            action_tree.add_line(line).unwrap();
        }
        for line in &removed_lines {
            action_tree.remove_line(line).unwrap();
        }

        let (tree_config, _, _, action_root) = action_tree.eject();

        // game instance
        let mut game = Self {
            state: Decode::decode(decoder)?,
            card_config: Decode::decode(decoder)?,
            tree_config,
            added_lines,
            removed_lines,
            action_root,
            num_combinations: Decode::decode(decoder)?,
            is_compression_enabled: Decode::decode(decoder)?,
            misc_memory_usage: Decode::decode(decoder)?,
            num_storage_actions: Decode::decode(decoder)?,
            num_storage_chances: Decode::decode(decoder)?,
            storage1: Decode::decode(decoder)?,
            storage2: Decode::decode(decoder)?,
            storage_chance: Decode::decode(decoder)?,
            storage1_compressed: Decode::decode(decoder)?,
            storage2_compressed: Decode::decode(decoder)?,
            storage_chance_compressed: Decode::decode(decoder)?,
            locking_strategy: Decode::decode(decoder)?,
            ..Default::default()
        };

        let root_cfvalue_ip = Vec::<f32>::decode(decoder)?;
        let history = Vec::<usize>::decode(decoder)?;
        let is_normalized_weight_cached = bool::decode(decoder)?;

        // store base pointers
        ACTION_BASE.with(|ps| {
            if game.state >= State::MemoryAllocated {
                let (base1, base2);
                if game.is_compression_enabled {
                    base1 = game.storage1_compressed.lock().as_mut_ptr() as *mut u8;
                    base2 = game.storage2_compressed.lock().as_mut_ptr() as *mut u8;
                } else {
                    base1 = game.storage1.lock().as_mut_ptr() as *mut u8;
                    base2 = game.storage2.lock().as_mut_ptr() as *mut u8;
                }
                ps.set((base1, base2));
            } else {
                ps.set((ptr::null_mut(), ptr::null_mut()));
            }
        });

        CHANCE_BASE.with(|p| {
            if game.state >= State::MemoryAllocated {
                let base = match game.is_compression_enabled {
                    true => game.storage_chance_compressed.lock().as_mut_ptr() as *mut u8,
                    false => game.storage_chance.lock().as_mut_ptr() as *mut u8,
                };
                p.set(base);
            } else {
                p.set(ptr::null_mut());
            }
        });

        // game tree
        game.node_arena = Decode::decode(decoder)?;

        // initialization
        if game.state >= State::TreeBuilt {
            game.init_hands();
            game.init_card_fields();
            game.init_interpreter();

            game.root_cfvalue_ip.copy_from_slice(&root_cfvalue_ip);
            game.apply_history(&history);
            if is_normalized_weight_cached {
                game.cache_normalized_weights();
            }
        }

        Ok(game)
    }
}

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
