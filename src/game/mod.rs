mod base;
mod evaluation;
mod interpreter;
mod node;

#[cfg(feature = "bincode")]
mod serialization;

#[cfg(test)]
mod tests;

use crate::action_tree::*;
use crate::card::*;
use crate::mutex_like::*;
use node::*;
use std::collections::BTreeMap;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
enum State {
    ConfigError = 0,
    #[default]
    Uninitialized = 1,
    TreeBuilt = 2,
    MemoryAllocated = 3,
    Solved = 4,
}

/// A struct representing a postflop game.
#[derive(Default)]
pub struct PostFlopGame {
    // state
    state: State,

    // postflop game configurations
    card_config: CardConfig,
    tree_config: TreeConfig,
    added_lines: Vec<Vec<Action>>,
    removed_lines: Vec<Vec<Action>>,
    action_root: Box<MutexLike<ActionTreeNode>>,

    // computed from configurations
    num_combinations: f64,
    initial_weights: [Vec<f32>; 2],
    private_cards: [Vec<(u8, u8)>; 2],
    same_hand_index: [Vec<u16>; 2],
    valid_indices_flop: [Vec<u16>; 2],
    valid_indices_turn: Vec<[Vec<u16>; 2]>,
    valid_indices_river: Vec<[Vec<u16>; 2]>,
    hand_strength: Vec<[Vec<StrengthItem>; 2]>,
    turn_isomorphism_ref: Vec<u8>,
    turn_isomorphism_card: Vec<u8>,
    turn_isomorphism_swap: [SwapList; 4],
    river_isomorphism_ref: Vec<Vec<u8>>,
    river_isomorphism_card: Vec<Vec<u8>>,
    river_isomorphism_swap: [[SwapList; 4]; 4],

    // store options
    storage_mode: BoardState,
    target_storage_mode: BoardState,
    num_nodes: [u64; 3],
    is_compression_enabled: bool,
    num_storage: u64,
    num_storage_ip: u64,
    num_storage_chance: u64,
    misc_memory_usage: u64,

    // global storage
    node_arena: Vec<MutexLike<PostFlopNode>>,
    storage1: Vec<u8>,
    storage2: Vec<u8>,
    storage_ip: Vec<u8>,
    storage_chance: Vec<u8>,
    locking_strategy: BTreeMap<usize, Vec<f32>>,

    // result interpreter
    history: Vec<usize>,
    is_normalized_weight_cached: bool,
    current_node_index: usize,
    turn: u8,
    river: u8,
    chance_factor: i32,
    turn_swapped_suit: Option<(u8, u8)>,
    turn_swap: Option<u8>,
    river_swap: Option<(u8, u8)>,
    total_bet_amount: [i32; 2],
    prev_bet_amount: i32,
    weights: [Vec<f32>; 2],
    normalized_weights: [Vec<f32>; 2],
    cfvalues_cache: [Vec<f32>; 2],
}

unsafe impl Send for PostFlopGame {}
unsafe impl Sync for PostFlopGame {}
