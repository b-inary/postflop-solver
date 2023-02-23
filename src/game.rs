use crate::action_tree::*;
use crate::card::*;
use crate::interface::*;
use crate::mutex_like::*;
use crate::node::*;
use crate::sliceop::*;
use crate::utility::*;
use std::collections::BTreeMap;
use std::mem::size_of;
use std::mem::{self, MaybeUninit};
use std::ptr;

#[cfg(feature = "bincode")]
use bincode::{
    error::{DecodeError, EncodeError},
    Decode, Encode,
};

#[cfg(feature = "custom-alloc")]
use std::vec;

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
    is_compression_enabled: bool,
    misc_memory_usage: u64,
    num_storage_actions: u64,
    num_storage_chances: u64,

    // global storage
    node_arena: Vec<MutexLike<PostFlopNode>>,
    storage1: MutexLike<Vec<f32>>,
    storage2: MutexLike<Vec<f32>>,
    storage_chance: MutexLike<Vec<f32>>,
    storage1_compressed: MutexLike<Vec<u16>>,
    storage2_compressed: MutexLike<Vec<i16>>,
    storage_chance_compressed: MutexLike<Vec<i16>>,
    locking_strategy: BTreeMap<usize, Vec<f32>>,

    // result interpreter
    root_cfvalue_ip: Vec<f32>,
    history: Vec<usize>,
    is_normalized_weight_cached: bool,
    node_ptr: *mut PostFlopNode,
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

struct BuildTreeInfo {
    flop_index: usize,
    turn_index: usize,
    river_index: usize,
    num_storage_actions: u64,
    num_storage_chances: u64,
}

#[inline]
fn min(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else {
        y
    }
}

/// Decodes the encoded `i16` slice to the `f32` slice.
#[inline]
fn decode_signed_slice(slice: &[i16], scale: f32) -> Vec<f32> {
    let decoder = scale / i16::MAX as f32;
    let mut result = Vec::<f32>::with_capacity(slice.len());
    let ptr = result.as_mut_ptr();
    unsafe {
        for i in 0..slice.len() {
            *ptr.add(i) = (*slice.get_unchecked(i)) as f32 * decoder;
        }
        result.set_len(slice.len());
    }
    result
}

impl Game for PostFlopGame {
    type Node = PostFlopNode;

    #[inline]
    fn root(&self) -> MutexGuardLike<Self::Node> {
        self.node_arena[0].lock()
    }

    #[inline]
    fn num_private_hands(&self, player: usize) -> usize {
        self.private_cards[player].len()
    }

    #[inline]
    fn initial_weights(&self, player: usize) -> &[f32] {
        &self.initial_weights[player]
    }

    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = (half_pot - rake) / self.num_combinations;
        let amount_lose = -half_pot / self.num_combinations;

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let mut cfreach_sum = 0.0;
        let mut cfreach_minus = [0.0; 52];

        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });

        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let valid_indices = if node.river != NOT_DEALT {
                &self.valid_indices_river[card_pair_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.valid_indices_turn[node.turn as usize]
            } else {
                &self.valid_indices_flop
            };

            let opponent_indices = &valid_indices[player ^ 1];
            for &i in opponent_indices {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(i as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let player_indices = &valid_indices[player];
            let same_hand_index = &self.same_hand_index[player];
            for &i in player_indices {
                unsafe {
                    let (c1, c2) = *player_cards.get_unchecked(i as usize);
                    let same_i = *same_hand_index.get_unchecked(i as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };
                    // inclusion-exclusion principle
                    let cfreach = cfreach_sum + cfreach_same
                        - *cfreach_minus.get_unchecked(c1 as usize)
                        - *cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(i as usize) = (payoff * cfreach) as f32;
                }
            }
        }
        // showdown (optimized for no rake; 2-pass)
        else if rake == 0.0 {
            let pair_index = card_pair_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let mut i = 1;

            for &StrengthItem { strength, index } in valid_player_strength {
                unsafe {
                    while opponent_strength.get_unchecked(i).strength < strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i += 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) = (amount_win * cfreach) as f32;
                }
            }

            cfreach_sum = 0.0;
            cfreach_minus.fill(0.0);
            i = opponent_strength.len() - 2;

            for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
                unsafe {
                    while opponent_strength.get_unchecked(i).strength > strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i -= 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) += (amount_lose * cfreach) as f32;
                }
            }
        }
        // showdown (raked; 3-pass)
        else {
            let amount_tie = -0.5 * rake / self.num_combinations;
            let same_hand_index = &self.same_hand_index[player];

            let pair_index = card_pair_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let valid_opponent_strength = &opponent_strength[1..opponent_strength.len() - 1];

            for &StrengthItem { index, .. } in valid_opponent_strength {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(index as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(index as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let mut cfreach_sum_win = 0.0;
            let mut cfreach_sum_tie = 0.0;
            let mut cfreach_minus_win = [0.0; 52];
            let mut cfreach_minus_tie = [0.0; 52];

            let mut i = 1;
            let mut j = 1;
            let mut prev_strength = 0; // strength is always > 0

            for &StrengthItem { strength, index } in valid_player_strength {
                unsafe {
                    if strength > prev_strength {
                        prev_strength = strength;

                        if i < j {
                            cfreach_sum_win = cfreach_sum_tie;
                            cfreach_minus_win = cfreach_minus_tie;
                            i = j;
                        }

                        while opponent_strength.get_unchecked(i).strength < strength {
                            let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_win += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c1 as usize) += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c2 as usize) += cfreach_i;
                            i += 1;
                        }

                        if j < i {
                            cfreach_sum_tie = cfreach_sum_win;
                            cfreach_minus_tie = cfreach_minus_win;
                            j = i;
                        }

                        while opponent_strength.get_unchecked(j).strength == strength {
                            let opponent_index = opponent_strength.get_unchecked(j).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_j = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_tie += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c1 as usize) += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c2 as usize) += cfreach_j;
                            j += 1;
                        }
                    }

                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach_total = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    let cfreach_win = cfreach_sum_win
                        - cfreach_minus_win.get_unchecked(c1 as usize)
                        - cfreach_minus_win.get_unchecked(c2 as usize);
                    let cfreach_tie = cfreach_sum_tie
                        - cfreach_minus_tie.get_unchecked(c1 as usize)
                        - cfreach_minus_tie.get_unchecked(c2 as usize);
                    let same_i = *same_hand_index.get_unchecked(index as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };

                    let cfvalue = amount_win * cfreach_win
                        + amount_tie * (cfreach_tie - cfreach_win + cfreach_same)
                        + amount_lose * (cfreach_total - cfreach_tie);
                    *result.get_unchecked_mut(index as usize) = cfvalue as f32;
                }
            }
        }
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.state == State::Solved
    }

    #[inline]
    fn set_solved(&mut self, cfvalue_ip: &[f32]) {
        self.state = State::Solved;
        self.root_cfvalue_ip.copy_from_slice(cfvalue_ip);
        let history = self.history.clone();
        self.apply_history(&history);
    }

    #[inline]
    fn is_raked(&self) -> bool {
        self.tree_config.rake_rate > 0.0 && self.tree_config.rake_cap > 0.0
    }

    #[inline]
    fn isomorphic_chances(&self, node: &Self::Node) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism_ref
        } else {
            &self.river_isomorphism_ref[node.turn as usize]
        }
    }

    #[inline]
    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(u16, u16)>; 2] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism_swap[self.turn_isomorphism_card[index] as usize & 3]
        } else {
            &self.river_isomorphism_swap[node.turn as usize & 3]
                [self.river_isomorphism_card[node.turn as usize][index] as usize & 3]
        }
    }

    #[inline]
    fn locking_strategy(&self, node: &Self::Node) -> &[f32] {
        if !node.is_locked {
            &[]
        } else {
            let offset = self.node_offset(node);
            self.locking_strategy.get(&offset).unwrap()
        }
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.state == State::MemoryAllocated
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }
}

impl PostFlopGame {
    /// Creates a new empty [`PostFlopGame`] (needs `update_config()` before solving).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`PostFlopGame`] with the specified configuration.
    #[inline]
    pub fn with_config(card_config: CardConfig, action_tree: ActionTree) -> Result<Self, String> {
        let mut game = Self::new();
        game.update_config(card_config, action_tree)?;
        Ok(game)
    }

    /// Updates the game configuration. The solved result will be lost.
    #[inline]
    pub fn update_config(
        &mut self,
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<(), String> {
        if !action_tree.invalid_terminals().is_empty() {
            return Err("Invalid terminal is found in action tree".to_string());
        }
        self.card_config = card_config;
        (
            self.tree_config,
            self.added_lines,
            self.removed_lines,
            self.action_root,
        ) = action_tree.eject();
        self.state = State::ConfigError;
        self.check_card_config()?;
        self.init()?;
        Ok(())
    }

    /// Obtains the card configuration.
    #[inline]
    pub fn card_config(&self) -> &CardConfig {
        &self.card_config
    }

    /// Obtains the tree configuration.
    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Obtains the added lines.
    #[inline]
    pub fn added_lines(&self) -> &[Vec<Action>] {
        &self.added_lines
    }

    /// Obtains the removed lines.
    #[inline]
    pub fn removed_lines(&self) -> &[Vec<Action>] {
        &self.removed_lines
    }

    /// Returns the card list of private hands of the given player.
    #[inline]
    pub fn private_cards(&self, player: usize) -> &[(u8, u8)] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        &self.private_cards[player]
    }

    /// Returns the estimated memory usage in bytes (uncompressed, compressed).
    #[inline]
    pub fn memory_usage(&self) -> (u64, u64) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        let num_elements = 2 * self.num_storage_actions + self.num_storage_chances;
        let uncompressed = self.misc_memory_usage + 4 * num_elements;
        let compressed = self.misc_memory_usage + 2 * num_elements;
        (uncompressed, compressed)
    }

    /// Remove lines after building the `PostFlopGame` but before allocating memory.
    ///
    /// This allows the removal of chance-specific lines (e.g., remove overbets on board-pairing
    /// turns) which we cannot do while building an action tree.
    // pub fn remove_lines(&mut self, lines: &[Vec<Action>]) -> Result<(), String> {
    //     if self.state <= State::Uninitialized {
    //         return Err("Game is not successfully initialized".to_string());
    //     } else if self.state >= State::MemoryAllocated {
    //         return Err("Game has already been allocated".to_string());
    //     }

    //     for line in lines {
    //         let mut root = self.root();
    //         let info = self.remove_line_recursive(&mut root, line)?;
    //         self.misc_memory_usage -= info.memory_usage_nodes;
    //         self.num_storage_actions -= info.num_storage_actions;
    //         self.num_storage_chances -= info.num_storage_chances;
    //     }

    //     Ok(())
    // }

    /// Allocates the memory.
    pub fn allocate_memory(&mut self, enable_compression: bool) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.state == State::MemoryAllocated && self.is_compression_enabled == enable_compression
        {
            return;
        }

        let bytes = if enable_compression { 2 } else { 4 };
        if bytes * self.num_storage_actions > isize::MAX as u64
            || bytes * self.num_storage_chances > isize::MAX as u64
        {
            panic!("Memory usage exceeds maximum size");
        }

        self.state = State::MemoryAllocated;
        self.is_compression_enabled = enable_compression;

        self.clear_storage();

        let num_actions = self.num_storage_actions as usize;
        let num_chances = self.num_storage_chances as usize;
        if enable_compression {
            self.storage1_compressed = MutexLike::new(vec![0; num_actions]);
            self.storage2_compressed = MutexLike::new(vec![0; num_actions]);
            self.storage_chance_compressed = MutexLike::new(vec![0; num_chances]);
        } else {
            self.storage1 = MutexLike::new(vec![0.0; num_actions]);
            self.storage2 = MutexLike::new(vec![0.0; num_actions]);
            self.storage_chance = MutexLike::new(vec![0.0; num_chances]);
        }

        let mut action_counter = 0;
        let mut chance_counter = 0;
        self.allocate_memory_recursive(&mut self.root(), &mut action_counter, &mut chance_counter);
    }

    /// Checks the card configuration.
    fn check_card_config(&mut self) -> Result<(), String> {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        if flop.contains(&NOT_DEALT) {
            return Err("Flop cards not initialized".to_string());
        }

        if flop.iter().any(|&c| 52 <= c) {
            return Err(format!("Flop cards must be in [0, 52): flop = {flop:?}"));
        }

        if flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2] {
            return Err(format!("Flop cards must be unique: flop = {flop:?}"));
        }

        if turn != NOT_DEALT {
            if 52 <= turn {
                return Err(format!("Turn card must be in [0, 52): turn = {turn}"));
            }

            if flop.contains(&turn) {
                return Err(format!(
                    "Turn card must be different from flop cards: turn = {turn}"
                ));
            }
        }

        if river != NOT_DEALT {
            if 52 <= river {
                return Err(format!("River card must be in [0, 52): river = {river}"));
            }

            if flop.contains(&river) {
                return Err(format!(
                    "River card must be different from flop cards: river = {river}"
                ));
            }

            if turn == river {
                return Err(format!(
                    "River card must be different from turn card: river = {river}"
                ));
            }

            if turn == NOT_DEALT {
                return Err(format!(
                    "River card specified without turn card: river = {river}"
                ));
            }
        }

        let expected_state = match (turn != NOT_DEALT, river != NOT_DEALT) {
            (false, _) => BoardState::Flop,
            (true, false) => BoardState::Turn,
            (true, true) => BoardState::River,
        };

        if self.tree_config.initial_state != expected_state {
            return Err(format!(
                "Invalid initial state of `tree_config`: expected = {:?}, actual = {:?}",
                expected_state, self.tree_config.initial_state
            ));
        }

        if range[0].is_empty() {
            return Err("OOP range is empty".to_string());
        }

        if range[1].is_empty() {
            return Err("IP range is empty".to_string());
        }

        self.init_hands();
        self.num_combinations = 0.0;

        for (&(c1, c2), &w1) in self.private_cards[0]
            .iter()
            .zip(self.initial_weights[0].iter())
        {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            for (&(c3, c4), &w2) in self.private_cards[1]
                .iter()
                .zip(self.initial_weights[1].iter())
            {
                let ip_mask: u64 = (1 << c3) | (1 << c4);
                if oop_mask & ip_mask == 0 {
                    self.num_combinations += w1 as f64 * w2 as f64;
                }
            }
        }

        if self.num_combinations == 0.0 {
            return Err("Valid card assignment does not exist".to_string());
        }

        Ok(())
    }

    /// Initializes fields `initial_weights` and `private_cards`.
    #[inline]
    fn init_hands(&mut self) {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if turn != NOT_DEALT {
            board_mask |= 1 << turn;
        }
        if river != NOT_DEALT {
            board_mask |= 1 << river;
        }

        for player in 0..2 {
            let (hands, weights) = range[player].get_hands_weights(board_mask);
            self.initial_weights[player] = weights;
            self.private_cards[player] = hands;
        }
    }

    /// Initializes the game.
    #[inline]
    fn init(&mut self) -> Result<(), String> {
        self.init_card_fields();
        self.init_root()?;
        self.init_interpreter();
        Ok(())
    }

    /// Initializes fields related to cards.
    fn init_card_fields(&mut self) {
        for player in 0..2 {
            let same_hand_index = &mut self.same_hand_index[player];
            same_hand_index.clear();

            let player_hands = &self.private_cards[player];
            let opponent_hands = &self.private_cards[player ^ 1];
            for hand in player_hands {
                same_hand_index.push(
                    opponent_hands
                        .binary_search(hand)
                        .map_or(u16::MAX, |i| i as u16),
                );
            }
        }

        (
            self.valid_indices_flop,
            self.valid_indices_turn,
            self.valid_indices_river,
        ) = self.card_config.valid_indices(&self.private_cards);

        self.hand_strength = self.card_config.hand_strength(&self.private_cards);

        (
            self.turn_isomorphism_ref,
            self.turn_isomorphism_card,
            self.turn_isomorphism_swap,
            self.river_isomorphism_ref,
            self.river_isomorphism_card,
            self.river_isomorphism_swap,
        ) = self.card_config.isomorphism(&self.private_cards);
    }

    /// Initializes the root node of game tree.
    fn init_root(&mut self) -> Result<(), String> {
        let num_nodes = self.count_num_nodes();
        let total_num_nodes = num_nodes[0] + num_nodes[1] + num_nodes[2];

        if total_num_nodes > u32::MAX as u64
            || size_of::<PostFlopNode>() as u64 * total_num_nodes > isize::MAX as u64
        {
            return Err("Too many nodes".to_string());
        }

        let mut info = BuildTreeInfo {
            flop_index: 0,
            turn_index: num_nodes[0] as usize,
            river_index: (num_nodes[0] + num_nodes[1]) as usize,
            num_storage_actions: 0,
            num_storage_chances: 0,
        };

        match self.tree_config.initial_state {
            BoardState::Flop => info.flop_index += 1,
            BoardState::Turn => info.turn_index += 1,
            BoardState::River => info.river_index += 1,
        }

        self.node_arena = (0..total_num_nodes)
            .map(|_| MutexLike::new(PostFlopNode::default()))
            .collect::<Vec<_>>();

        let mut root = self.node_arena[0].lock();
        root.turn = self.card_config.turn;
        root.river = self.card_config.river;

        self.build_tree_recursive(0, &self.action_root.lock(), &mut info);

        self.misc_memory_usage = self.memory_usage_internal();
        self.num_storage_actions = info.num_storage_actions;
        self.num_storage_chances = info.num_storage_chances;

        self.clear_storage();
        self.state = State::TreeBuilt;

        Ok(())
    }

    /// Initializes the interpreter.
    #[inline]
    fn init_interpreter(&mut self) {
        let vecs = [
            vec![0.0; self.num_private_hands(0)],
            vec![0.0; self.num_private_hands(1)],
        ];

        self.root_cfvalue_ip = vec![0.0; self.num_private_hands(PLAYER_IP as usize)];
        self.weights = vecs.clone();
        self.normalized_weights = vecs.clone();
        self.cfvalues_cache = vecs;

        self.back_to_root();
    }

    /// Clears the storage.
    #[inline]
    fn clear_storage(&self) {
        self.storage1.lock().clear();
        self.storage2.lock().clear();
        self.storage_chance.lock().clear();
        self.storage1_compressed.lock().clear();
        self.storage2_compressed.lock().clear();
        self.storage_chance_compressed.lock().clear();
        self.storage1.lock().shrink_to_fit();
        self.storage2.lock().shrink_to_fit();
        self.storage_chance.lock().shrink_to_fit();
        self.storage1_compressed.lock().shrink_to_fit();
        self.storage2_compressed.lock().shrink_to_fit();
        self.storage_chance_compressed.lock().shrink_to_fit();
    }

    /// Counts the number of nodes in the game tree.
    #[inline]
    fn count_num_nodes(&self) -> [u64; 3] {
        let (turn_coef, river_coef) = match (self.card_config.turn, self.card_config.river) {
            (NOT_DEALT, _) => {
                let mut river_coef = 0;
                let flop = self.card_config.flop;
                let skip_cards = &self.turn_isomorphism_card;
                let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
                let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();
                for turn in 0..52 {
                    if (1 << turn) & (flop_mask | skip_mask) == 0 {
                        river_coef += 48 - self.river_isomorphism_card[turn].len();
                    }
                }
                (49 - self.turn_isomorphism_card.len(), river_coef)
            }
            (turn, NOT_DEALT) => (1, 48 - self.river_isomorphism_card[turn as usize].len()),
            _ => (0, 1),
        };

        let num_action_nodes = count_num_action_nodes(&self.action_root.lock());

        [
            num_action_nodes[0],
            num_action_nodes[1] * turn_coef as u64,
            num_action_nodes[2] * river_coef as u64,
        ]
    }

    /// Computes the memory usage of this struct.
    #[inline]
    fn memory_usage_internal(&self) -> u64 {
        // untracked: tree_config, action_root

        let mut memory_usage = mem::size_of::<Self>() as u64;

        memory_usage += vec_memory_usage(&self.added_lines);
        memory_usage += vec_memory_usage(&self.removed_lines);
        for line in &self.added_lines {
            memory_usage += vec_memory_usage(line);
        }
        for line in &self.removed_lines {
            memory_usage += vec_memory_usage(line);
        }

        memory_usage += vec_memory_usage(&self.valid_indices_turn);
        memory_usage += vec_memory_usage(&self.valid_indices_river);
        memory_usage += vec_memory_usage(&self.hand_strength);
        memory_usage += vec_memory_usage(&self.turn_isomorphism_ref);
        memory_usage += vec_memory_usage(&self.turn_isomorphism_card);
        memory_usage += vec_memory_usage(&self.river_isomorphism_ref);
        memory_usage += vec_memory_usage(&self.river_isomorphism_card);

        for player in 0..2 {
            memory_usage += vec_memory_usage(&self.initial_weights[player]);
            memory_usage += vec_memory_usage(&self.private_cards[player]);
            memory_usage += vec_memory_usage(&self.same_hand_index[player]);
            memory_usage += vec_memory_usage(&self.valid_indices_flop[player]);
            for indices in &self.valid_indices_turn {
                memory_usage += vec_memory_usage(&indices[player]);
            }
            for indices in &self.valid_indices_river {
                memory_usage += vec_memory_usage(&indices[player]);
            }
            for strength in &self.hand_strength {
                memory_usage += vec_memory_usage(&strength[player]);
            }
            for swap in &self.turn_isomorphism_swap {
                memory_usage += vec_memory_usage(&swap[player]);
            }
            for swap_list in &self.river_isomorphism_swap {
                for swap in swap_list {
                    memory_usage += vec_memory_usage(&swap[player]);
                }
            }
        }

        memory_usage += vec_memory_usage(&self.node_arena);

        memory_usage
    }

    /// Builds the game tree recursively.
    fn build_tree_recursive(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let mut node = self.node_arena[node_index].lock();
        node.player = action_node.player;
        node.amount = action_node.amount;

        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            self.push_chances(node_index, info);
            for action_index in 0..node.num_actions() {
                let child_index = node_index + node.children_offset as usize + action_index;
                self.build_tree_recursive(child_index, &action_node.children[0].lock(), info);
            }
        } else {
            self.push_actions(node_index, action_node, info);
            for action_index in 0..node.num_actions() {
                let child_index = node_index + node.children_offset as usize + action_index;
                self.build_tree_recursive(
                    child_index,
                    &action_node.children[action_index].lock(),
                    info,
                );
            }
        }
    }

    /// Pushes the chance actions to the `node`.
    fn push_chances(&self, node_index: usize, info: &mut BuildTreeInfo) {
        let mut node = self.node_arena[node_index].lock();
        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        // deal turn
        if node.turn == NOT_DEALT {
            let skip_cards = &self.turn_isomorphism_card;
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();

            node.children_offset = (info.turn_index - node_index) as u32;
            for card in 0..52 {
                if (1 << card) & (flop_mask | skip_mask) == 0 {
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = card;
                }
            }

            info.turn_index += node.num_children as usize;
        }
        // deal river
        else {
            let turn_mask = flop_mask | (1 << node.turn);
            let skip_cards = &self.river_isomorphism_card[node.turn as usize];
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();

            node.children_offset = (info.river_index - node_index) as u32;
            for card in 0..52 {
                if (1 << card) & (turn_mask | skip_mask) == 0 {
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = node.turn;
                    child.river = card;
                }
            }

            info.river_index += node.num_children as usize;
        }

        node.num_elements = match node.cfvalue_storage(PLAYER_OOP as usize) {
            CfValueStorage::None => 0,
            CfValueStorage::Sum => self.num_private_hands(PLAYER_OOP as usize),
            CfValueStorage::All => node.num_actions() * self.num_private_hands(PLAYER_OOP as usize),
        } as u32;

        node.num_elements_aux = match node.cfvalue_storage(PLAYER_IP as usize) {
            CfValueStorage::None => 0,
            CfValueStorage::Sum => self.num_private_hands(PLAYER_IP as usize),
            CfValueStorage::All => node.num_actions() * self.num_private_hands(PLAYER_IP as usize),
        } as u32;

        info.num_storage_chances += (node.num_elements + node.num_elements_aux) as u64;
    }

    /// Pushes the actions to the `node`.
    fn push_actions(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let mut node = self.node_arena[node_index].lock();

        let street = match (node.turn, node.river) {
            (NOT_DEALT, _) => BoardState::Flop,
            (_, NOT_DEALT) => BoardState::Turn,
            _ => BoardState::River,
        };

        let base = match street {
            BoardState::Flop => &mut info.flop_index,
            BoardState::Turn => &mut info.turn_index,
            BoardState::River => &mut info.river_index,
        };

        node.children_offset = (*base - node_index) as u32;
        node.num_children = action_node.children.len() as u32;
        *base += node.num_children as usize;

        for (child, action) in node.children().iter().zip(action_node.actions.iter()) {
            let mut child = child.lock();
            child.prev_action = *action;
            child.turn = node.turn;
            child.river = node.river;
        }

        let num_private_hands = self.num_private_hands(node.player as usize);
        node.num_elements = (node.num_actions() * num_private_hands) as u32;

        info.num_storage_actions += node.num_elements as u64;
    }

    /// Reverse the action of push_chances
    // fn reverse_push_chances(&self, node: &PostFlopNode, info: &mut BuildTreeInfo) {
    //     info.memory_usage_nodes += vec_memory_usage(&node.actions);
    //     info.memory_usage_nodes += vec_memory_usage(&node.children);
    //     info.num_storage_chances += (node.num_elements + node.num_elements_aux) as u64;
    // }

    // fn reverse_push_actions(&self, node: &PostFlopNode, info: &mut BuildTreeInfo) {
    //     info.memory_usage_nodes += vec_memory_usage(&node.actions);
    //     info.memory_usage_nodes += vec_memory_usage(&node.children);
    //     info.num_storage_actions += node.num_elements as u64;
    // }

    // fn calculate_removed_line_info_recursive(&self, node: &PostFlopNode, info: &mut BuildTreeInfo) {
    //     if node.is_terminal() {
    //         return;
    //     }

    //     if node.is_chance() {
    //         self.reverse_push_chances(node, info);
    //     } else {
    //         self.reverse_push_actions(node, info);
    //     }

    //     for action in node.action_indices() {
    //         self.calculate_removed_line_info_recursive(&node.play(action), info);
    //     }
    // }

    /// Recursively remove a line from a PostFlopGame tree
    // fn remove_line_recursive(
    //     &self,
    //     node: &mut PostFlopNode,
    //     line: &[Action],
    // ) -> Result<BuildTreeInfo, String> {
    //     if line.is_empty() {
    //         return Err("Empty line".to_string());
    //     }

    //     if node.is_terminal() {
    //         return Err("Unexpected terminal node".to_string());
    //     }

    //     let action = line[0];
    //     let search_result = node.actions.binary_search(&action);
    //     if search_result.is_err() {
    //         return Err(format!("Action does not exist: {action:?}"));
    //     }

    //     let index = search_result.unwrap();
    //     if line.len() > 1 {
    //         let result = self.remove_line_recursive(&mut node.children[index].lock(), &line[1..]);
    //         return result;
    //     }

    //     if node.is_chance() {
    //         return Err("Cannot remove a line ending in a chance action".to_string());
    //     }

    //     if node.num_actions() <= 1 {
    //         return Err("Cannot remove the last action from a node".to_string());
    //     }

    //     // Remove action/children at index. To do this we must
    //     // 1. compute the storage space required by the tree rooted at action index
    //     // 2. remove children and actions
    //     // 3. re-define `num_elements` after we remove children and actions

    //     // STEP 1
    //     let mut info = BuildTreeInfo {
    //         memory_usage_nodes: (mem::size_of::<Action>() + mem::size_of::<PostFlopNode>()) as u64,
    //         num_storage_actions: self.num_private_hands(node.player as usize) as u64,
    //         num_storage_chances: 0,
    //     };

    //     let node_to_remove = node.play(index);
    //     self.calculate_removed_line_info_recursive(&node_to_remove, &mut info);

    //     // STEP 2
    //     node.actions.remove(index);
    //     node.children.remove(index);

    //     node.actions.shrink_to_fit();
    //     node.children.shrink_to_fit();

    //     // STEP 3
    //     node.num_elements -= self.num_private_hands(node.player as usize);

    //     Ok(info)
    // }

    /// Allocates memory recursively.
    fn allocate_memory_recursive(
        &self,
        node: &mut PostFlopNode,
        action_counter: &mut usize,
        chance_counter: &mut usize,
    ) {
        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            if node.num_elements > 0 {
                unsafe {
                    if self.is_compression_enabled {
                        let ptr = self.storage_chance_compressed.lock().as_mut_ptr();
                        node.storage1 = ptr.add(*chance_counter) as *mut u8;
                    } else {
                        let ptr = self.storage_chance.lock().as_mut_ptr();
                        node.storage1 = ptr.add(*chance_counter) as *mut u8;
                    }
                }
                *chance_counter += node.num_elements as usize;
            }
            if node.num_elements_aux > 0 {
                unsafe {
                    if self.is_compression_enabled {
                        let ptr = self.storage_chance_compressed.lock().as_mut_ptr();
                        node.storage2 = ptr.add(*chance_counter) as *mut u8;
                    } else {
                        let ptr = self.storage_chance.lock().as_mut_ptr();
                        node.storage2 = ptr.add(*chance_counter) as *mut u8;
                    }
                }
                *chance_counter += node.num_elements_aux as usize;
            }
        } else {
            unsafe {
                if self.is_compression_enabled {
                    let ptr1 = self.storage1_compressed.lock().as_mut_ptr();
                    let ptr2 = self.storage2_compressed.lock().as_mut_ptr();
                    node.storage1 = ptr1.add(*action_counter) as *mut u8;
                    node.storage2 = ptr2.add(*action_counter) as *mut u8;
                } else {
                    let ptr1 = self.storage1.lock().as_mut_ptr();
                    let ptr2 = self.storage2.lock().as_mut_ptr();
                    node.storage1 = ptr1.add(*action_counter) as *mut u8;
                    node.storage2 = ptr2.add(*action_counter) as *mut u8;
                }
            }
            *action_counter += node.num_elements as usize;
        }

        for action in node.action_indices() {
            self.allocate_memory_recursive(&mut node.play(action), action_counter, chance_counter);
        }
    }

    /// Moves the current node back to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.history.clear();
        self.is_normalized_weight_cached = false;
        self.node_ptr = &mut *self.root();
        self.turn = self.card_config.turn;
        self.river = self.card_config.river;
        self.chance_factor = 1;
        self.turn_swapped_suit = None;
        self.turn_swap = None;
        self.river_swap = None;
        self.total_bet_amount = [0, 0];
        self.prev_bet_amount = 0;

        self.weights[0].copy_from_slice(&self.initial_weights[0]);
        self.weights[1].copy_from_slice(&self.initial_weights[1]);
        self.cfvalues_cache[1].copy_from_slice(&self.root_cfvalue_ip);
    }

    /// Returns the history of the current node.
    ///
    /// The history is a list of action indices, i.e., the arguments of [`play`]. If `usize::MAX`
    /// was passed to [`play`], it is replaced with the actual action index.
    ///
    /// [`play`]: #method.play
    #[inline]
    pub fn history(&self) -> &[usize] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        &self.history
    }

    /// Applies the given history from the root node.
    ///
    /// This method first calls [`back_to_root`] and then calls [`play`] for each action in the
    /// history. The action of `usize::MAX` is allowed for chance nodes.
    ///
    /// [`back_to_root`]: #method.back_to_root
    /// [`play`]: #method.play
    #[inline]
    pub fn apply_history(&mut self, history: &[usize]) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.back_to_root();
        for &action in history {
            self.play(action);
        }
    }

    /// Returns whether the current node is a terminal node.
    ///
    /// Note that the turn/river node after the call action after the all-in action is considered
    /// terminal.
    #[inline]
    pub fn is_terminal_node(&self) -> bool {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        let node = self.node();
        node.is_terminal() || node.amount == self.tree_config.effective_stack
    }

    /// Returns whether the current node is a chance node (i.e., turn/river node).
    ///
    /// Note that the terminal node is not considered a chance node.
    #[inline]
    pub fn is_chance_node(&self) -> bool {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.node().is_chance() && !self.is_terminal_node()
    }

    /// Returns the available actions for the current node.
    ///
    /// If the current node is a terminal, returns an empty list. If the current node is a
    /// turn/river node and not a terminal, isomorphic chances are grouped into one representative
    /// action (in most cases, you should use the [`possible_cards`] method).
    ///
    /// [`possible_cards`]: #method.possible_cards
    #[inline]
    pub fn available_actions(&self) -> Vec<Action> {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.is_terminal_node() {
            Vec::new()
        } else {
            self.node()
                .children()
                .iter()
                .map(|c| c.lock().prev_action)
                .collect()
        }
    }

    /// If the current node is a chance node, returns a list of cards that may be dealt.
    ///
    /// The returned value is a 64-bit integer.
    /// The `i`-th bit is set to 1 if the card of ID `i` may be dealt.
    /// If the current node is not a chance node, returns `0`.
    ///
    /// Card ID: `"2c"` => `0`, `"2d"` => `1`, `"2h"` => `2`, ..., `"As"` => `51`.
    pub fn possible_cards(&self) -> u64 {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if !self.is_chance_node() {
            return 0;
        }

        let flop = self.card_config.flop;
        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }

        let mut mask: u64 = (1 << 52) - 1;

        for &(c1, c2) in &self.private_cards[0] {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            if board_mask & oop_mask == 0 {
                for &(c3, c4) in &self.private_cards[1] {
                    let ip_mask: u64 = (1 << c3) | (1 << c4);
                    if (board_mask | oop_mask) & ip_mask == 0 {
                        mask &= oop_mask | ip_mask;
                    }
                }
                if mask == 0 {
                    break;
                }
            }
        }

        ((1 << 52) - 1) ^ (board_mask | mask)
    }

    /// Returns the current player (0 = OOP, 1 = IP).
    ///
    /// If the current node is a terminal node or a chance node, returns an undefined value.
    #[inline]
    pub fn current_player(&self) -> usize {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.node().player()
    }

    /// Returns the current board.
    ///
    /// The returned vector is of length 3, 4, or 5. The flop cards, the turn card, and the river
    /// card, if any, are stored in this order.
    #[inline]
    pub fn current_board(&self) -> Vec<u8> {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        let mut ret = self.card_config.flop.to_vec();
        if self.turn != NOT_DEALT {
            ret.push(self.turn);
        }
        if self.river != NOT_DEALT {
            ret.push(self.river);
        }
        ret
    }

    /// Plays the given action. Playing an action from a terminal node is not allowed.
    ///
    /// - `action`
    ///   - If the current node is a chance node, the `action` corresponds to the card ID of the
    ///     dealt card. The `action` can be `usize::MAX`, in which case the actual card is chosen
    ///     from possible cards.
    ///   - If the current node is not a chance node, plays the `action`-th action of
    ///     [`available_actions`].
    ///
    /// Panics if the memory is not yet allocated or the current node is a terminal node.
    ///
    /// [`available_actions`]: #method.available_actions
    pub fn play(&mut self, action: usize) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        // chance node
        if self.is_chance_node() {
            let is_turn = self.turn == NOT_DEALT;
            let actual_card = if action == usize::MAX {
                self.possible_cards().trailing_zeros() as u8
            } else {
                action as u8
            };

            // swap the suit if swapping was performed in turn
            let action_card = if let Some((suit1, suit2)) = self.turn_swapped_suit {
                if actual_card & 3 == suit1 {
                    actual_card - suit1 + suit2
                } else if actual_card & 3 == suit2 {
                    actual_card + suit1 - suit2
                } else {
                    actual_card
                }
            } else {
                actual_card
            };

            let actions = self.available_actions();
            let mut action_index = usize::MAX;

            // find the action index from available actions
            for (i, &action) in actions.iter().enumerate() {
                if action == Action::Chance(action_card) {
                    action_index = i;
                    break;
                }
            }

            // find the action index from isomorphic chances
            if action_index == usize::MAX {
                let isomorphism = self.isomorphic_chances(self.node());
                let isomorphic_cards = self.isomorphic_cards(self.node());
                for (i, &repr_index) in isomorphism.iter().enumerate() {
                    if action_card == isomorphic_cards[i] {
                        action_index = repr_index as usize;
                        if is_turn {
                            if let Action::Chance(repr_card) = actions[repr_index as usize] {
                                self.turn_swapped_suit = Some((action_card & 3, repr_card & 3));
                            }
                            self.turn_swap = Some(action_card & 3);
                        } else {
                            // `self.turn != self.node().turn` if `self.turn_swap.is_some()`.
                            // This is possible only when the flop is monotone.
                            // In this case, there is only one suit that can be swapped and the
                            // following code works correctly.
                            self.river_swap = Some((
                                self.turn,
                                self.river_isomorphism_card[self.turn as usize][i] & 3,
                            ));
                        }
                        break;
                    }
                }
            }

            // panic if the action is not found
            if action_index == usize::MAX {
                panic!("Invalid action");
            }

            // update the weights
            for player in 0..2 {
                self.weights[player]
                    .iter_mut()
                    .zip(self.private_cards[player].iter())
                    .for_each(|(w, &(c1, c2))| {
                        if c1 == actual_card || c2 == actual_card {
                            *w = 0.0;
                        }
                    });
            }

            // cache the counterfactual values
            let player_ip = PLAYER_IP as usize;
            let num_hands = self.num_private_hands(player_ip);
            let vec = if self.is_compression_enabled {
                let src = self.node().cfvalues_chance_compressed(player_ip);
                let slice = row(src, action_index, num_hands);
                let scale = self.node().cfvalue_chance_scale(player_ip);
                decode_signed_slice(slice, scale)
            } else {
                let src = self.node().cfvalues_chance(player_ip);
                row(src, action_index, num_hands).to_vec()
            };
            self.cfvalues_cache[player_ip].copy_from_slice(&vec);

            // update the state
            self.node_ptr = &mut *self.node().play(action_index);
            if is_turn {
                self.turn = actual_card;
                self.chance_factor *= 45;
            } else {
                self.river = actual_card;
                self.chance_factor *= 44;
            }
        }
        // player node
        else {
            // panic if the action is invalid
            if action >= self.node().num_actions() {
                panic!("Invalid action");
            }

            let player = self.node().player();
            let num_hands = self.num_private_hands(player);

            // update the weights
            if self.node().num_actions() > 1 {
                let strategy = self.strategy();
                let weights = row(&strategy, action, num_hands);
                mul_slice(&mut self.weights[player], weights);
            }

            // cache the counterfactual values
            let vec = if self.is_compression_enabled {
                let slice = row(self.node().cfvalues_compressed(), action, num_hands);
                let scale = self.node().cfvalue_scale();
                decode_signed_slice(slice, scale)
            } else {
                row(self.node().cfvalues(), action, num_hands).to_vec()
            };
            self.cfvalues_cache[player].copy_from_slice(&vec);

            // update the bet amounts
            match self.node().play(action).prev_action {
                Action::Call => {
                    self.total_bet_amount[player] = self.total_bet_amount[player ^ 1];
                    self.prev_bet_amount = 0;
                }
                Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => {
                    let to_call = self.total_bet_amount[player ^ 1] - self.total_bet_amount[player];
                    self.total_bet_amount[player] += amount - self.prev_bet_amount + to_call;
                    self.prev_bet_amount = amount;
                }
                _ => {}
            }

            // update the node
            self.node_ptr = &mut *self.node().play(action);
        }

        self.history.push(action);
        self.is_normalized_weight_cached = false;
    }

    /// Computes the normalized weights and caches them.
    ///
    /// After mutating the current node, this method must be called once before calling
    /// [`normalized_weights`], [`equity`], [`expected_values`], or [`expected_values_detail`].
    ///
    /// [`normalized_weights`]: #method.normalized_weights
    /// [`equity`]: #method.equity
    /// [`expected_values`]: #method.expected_values
    /// [`expected_values_detail`]: #method.expected_values_detail
    pub fn cache_normalized_weights(&mut self) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.is_normalized_weight_cached {
            return;
        }

        let mut board_mask: u64 = 0;
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }
        if self.river != NOT_DEALT {
            board_mask |= 1 << self.river;
        }

        let mut weight_sum = [0.0; 2];
        let mut weight_sum_minus = [[0.0; 52]; 2];

        for player in 0..2 {
            let weight_sum_player = &mut weight_sum[player];
            let weight_sum_minus_player = &mut weight_sum_minus[player];
            self.private_cards[player]
                .iter()
                .zip(self.weights[player].iter())
                .for_each(|(&(c1, c2), &w)| {
                    let mask: u64 = (1 << c1) | (1 << c2);
                    if mask & board_mask == 0 {
                        let w = w as f64;
                        *weight_sum_player += w;
                        weight_sum_minus_player[c1 as usize] += w;
                        weight_sum_minus_player[c2 as usize] += w;
                    }
                });
        }

        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let same_hand_index = &self.same_hand_index[player];
            let player_weights = &self.weights[player];
            let opponent_weights = &self.weights[player ^ 1];
            let opponent_weight_sum = weight_sum[player ^ 1];
            let opponent_weight_sum_minus = &weight_sum_minus[player ^ 1];

            self.normalized_weights[player]
                .iter_mut()
                .enumerate()
                .for_each(|(i, w)| {
                    let (c1, c2) = player_cards[i];
                    let mask: u64 = (1 << c1) | (1 << c2);
                    if mask & board_mask == 0 {
                        let same_i = same_hand_index[i];
                        let opponent_weight_same = if same_i == u16::MAX {
                            0.0
                        } else {
                            opponent_weights[same_i as usize] as f64
                        };
                        let opponent_weight = opponent_weight_sum + opponent_weight_same
                            - opponent_weight_sum_minus[c1 as usize]
                            - opponent_weight_sum_minus[c2 as usize];
                        *w = player_weights[i] * opponent_weight as f32;
                    } else {
                        *w = 0.0;
                    }
                });
        }

        self.is_normalized_weight_cached = true;
    }

    /// Returns the weights of each private hand of the given player.
    ///
    /// If a hand overlaps with the board, returns 0.0.
    #[inline]
    pub fn weights(&self, player: usize) -> &[f32] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        &self.weights[player]
    }

    /// Returns the normalized weights of each private hand of the given player.
    ///
    /// The "normalized weights" represent the actual number of combinations that the player is
    /// holding each hand.
    ///
    /// After mutating the current node, you must call the [`cache_normalized_weights`] method
    /// before calling this method.
    ///
    /// [`cache_normalized_weights`]: #method.cache_normalized_weights
    #[inline]
    pub fn normalized_weights(&self, player: usize) -> &[f32] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        &self.normalized_weights[player]
    }

    /// Returns the equity of each private hand of the given player.
    ///
    /// After mutating the current node, you must call the [`cache_normalized_weights`] method
    /// before calling this method.
    ///
    /// [`cache_normalized_weights`]: #method.cache_normalized_weights
    pub fn equity(&self, player: usize) -> Vec<f32> {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let num_hands = self.num_private_hands(player);
        let mut tmp = vec![0.0; num_hands];

        if self.river != NOT_DEALT {
            self.equity_internal(&mut tmp, player, self.turn, self.river, 0.5);
        } else if self.turn != NOT_DEALT {
            for river in 0..52 {
                if self.turn != river {
                    self.equity_internal(&mut tmp, player, self.turn, river, 0.5 / 44.0);
                }
            }
        } else {
            for turn in 0..52 {
                for river in turn + 1..52 {
                    self.equity_internal(&mut tmp, player, turn, river, 1.0 / (45.0 * 44.0));
                }
            }
        }

        tmp.iter()
            .zip(self.weights[player].iter())
            .zip(self.normalized_weights[player].iter())
            .map(|((&v, &w_raw), &w_normalized)| {
                if w_normalized > 0.0 {
                    v as f32 * (w_raw / w_normalized) + 0.5
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Returns the expected values of each private hand of the given player.
    ///
    /// Panics if the game is not solved.
    ///
    /// After mutating the current node, you must call the [`cache_normalized_weights`] method
    /// before calling this method.
    ///
    /// [`cache_normalized_weights`]: #method.cache_normalized_weights
    pub fn expected_values(&self, player: usize) -> Vec<f32> {
        if self.state != State::Solved {
            panic!("Game is not solved");
        }

        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let expected_value_detail = self.expected_values_detail(player);

        if self.is_terminal_node() || self.is_chance_node() || self.current_player() != player {
            return expected_value_detail;
        }

        let num_actions = self.node().num_actions();
        let num_hands = self.num_private_hands(player);
        let strategy = self.strategy();

        let mut ret = Vec::with_capacity(num_hands);
        for i in 0..num_hands {
            let mut expected_value = 0.0;
            for j in 0..num_actions {
                let index = i + j * num_hands;
                expected_value += expected_value_detail[index] * strategy[index];
            }
            ret.push(expected_value);
        }

        ret
    }

    /// Returns the expected values of each action of each private hand of the given player.
    ///
    /// If the given player is the current player, the return value is a vector of the length of
    /// `#(actions) * #(private hands)`. The expected value of the `i`-th action with the `j`-th
    /// private hand is stored in the `i * #(private hands) + j`-th element.
    ///
    /// Otherwise, this method is the same as the [`expected_values`] method, so the return vector
    /// is the length of `#(private hands)`.
    ///
    /// Panics if the game is not solved.
    ///
    /// After mutating the current node, you must call the [`cache_normalized_weights`] method
    /// before calling this method.
    ///
    /// [`expected_values`]: #method.expected_value
    /// [`cache_normalized_weights`]: #method.cache_normalized_weights
    pub fn expected_values_detail(&self, player: usize) -> Vec<f32> {
        if self.state != State::Solved {
            panic!("Game is not solved");
        }

        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let num_hands = self.num_private_hands(player);

        let mut have_actions = false;
        let mut normalizer = (self.num_combinations * self.chance_factor as f64) as f32;

        let mut ret = if self.node().is_terminal() {
            normalizer = self.num_combinations as f32;
            let mut ret = Vec::with_capacity(num_hands);
            let mut cfreach = self.weights[player ^ 1].clone();
            self.apply_swap(&mut cfreach, player ^ 1, true);
            self.evaluate(ret.spare_capacity_mut(), self.node(), player, &cfreach);
            unsafe { ret.set_len(num_hands) };
            ret
        } else if self.node().is_chance() {
            self.cfvalues_chance(player)
        } else if player != self.current_player() {
            self.cfvalues_cache[player].to_vec()
        } else if self.is_compression_enabled {
            have_actions = true;
            let slice = self.node().cfvalues_compressed();
            let scale = self.node().cfvalue_scale();
            decode_signed_slice(slice, scale)
        } else {
            have_actions = true;
            self.node().cfvalues().to_vec()
        };

        let starting_pot = self.tree_config.starting_pot;
        let total_bet_amount = self.total_bet_amount();
        let bias = (total_bet_amount[player] - total_bet_amount[player ^ 1]).max(0);

        ret.chunks_exact_mut(num_hands)
            .enumerate()
            .for_each(|(action, row)| {
                let is_fold = have_actions && self.node().play(action).prev_action == Action::Fold;
                self.apply_swap(row, player, false);
                row.iter_mut()
                    .zip(self.weights[player].iter())
                    .zip(self.normalized_weights[player].iter())
                    .for_each(|((v, &w_raw), &w_normalized)| {
                        if is_fold || w_normalized == 0.0 {
                            *v = 0.0;
                        } else {
                            *v *= normalizer * (w_raw / w_normalized);
                            *v += starting_pot as f32 * 0.5 + (self.node().amount + bias) as f32;
                        }
                    });
            });

        ret
    }

    /// Returns the strategy of the current player.
    ///
    /// The return value is a vector of the length of `#(actions) * #(private hands)`.
    /// The probability of the `i`-th action with the `j`-th private hand is stored in the
    /// `i * #(private hands) + j`-th element.
    ///
    /// If a hand overlaps with the board, an undefined value is returned.
    ///
    /// Panics if the current node is a terminal node or a chance node. Also, panics if the memory
    /// is not yet allocated.
    pub fn strategy(&self) -> Vec<f32> {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let player = self.current_player();
        let num_actions = self.node().num_actions();
        let num_hands = self.num_private_hands(player);

        let mut ret = if self.is_compression_enabled {
            normalized_strategy_compressed(self.node().strategy_compressed(), num_actions)
        } else {
            normalized_strategy(self.node().strategy(), num_actions)
        };

        let locking = self.locking_strategy(self.node());
        apply_locking_strategy(&mut ret, locking);

        ret.chunks_exact_mut(num_hands).for_each(|chunk| {
            self.apply_swap(chunk, player, false);
        });

        ret
    }

    /// Returns the total bet amount of each player (OOP, IP).
    #[inline]
    pub fn total_bet_amount(&self) -> [i32; 2] {
        self.total_bet_amount
    }

    /// Locks the strategy of the current node.
    ///
    /// The `strategy` argument must be a slice of the length of `#(actions) * #(private hands)`.
    ///
    /// - A negative value is treated as a zero.
    /// - If the `i * #(private hands) + j`-th element of the `strategy` is positive for some `i`,
    ///   the `j`-th private hand will be locked. The probability for each action will be normalized
    ///   so that their sum is 1.0.
    /// - If the `i * #(private hands) + j`-th element of the `strategy` is not positive for all
    ///   `i`, the `j`-th private hand will not be locked. That is, the solver can adjust the
    ///   strategy of the `j`-th private hand.
    pub fn lock_current_strategy(&mut self, strategy: &[f32]) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let player = self.current_player();
        let num_actions = self.node().num_actions();
        let num_hands = self.num_private_hands(player);

        if strategy.len() != num_actions * num_hands {
            panic!("Invalid strategy length");
        }

        let mut locking = vec![-1.0; num_actions * num_hands];

        for hand in 0..num_hands {
            let mut sum = 0.0;
            let mut lock = false;

            for action in 0..num_actions {
                let freq = strategy[action * num_hands + hand];
                if freq > 0.0 {
                    sum += freq as f64;
                    lock = true;
                }
            }

            if lock {
                for action in 0..num_actions {
                    let freq = strategy[action * num_hands + hand].max(0.0) as f64;
                    locking[action * num_hands + hand] = (freq / sum) as f32;
                }
            }
        }

        locking.chunks_exact_mut(num_hands).for_each(|chunk| {
            self.apply_swap(chunk, player, true);
        });

        let offset = self.node_offset(self.node());
        self.locking_strategy.insert(offset, locking);
        self.node_mut().is_locked = true;
    }

    /// Unlocks the strategy of the current node.
    #[inline]
    pub fn unlock_current_strategy(&mut self) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        if !self.node().is_locked {
            return;
        }

        let offset = self.node_offset(self.node());
        self.locking_strategy.remove(&offset);
        self.node_mut().is_locked = false;
    }

    /// Returns the locking strategy of the current node.
    ///
    /// If the current node is not locked, `None` is returned.
    ///
    /// Otherwise, returns a reference to the vector of the length of
    /// `#(actions) * #(private hands)`.
    /// The probability of the `i`-th action with the `j`-th private hand is stored in the
    /// `i * #(private hands) + j`-th element.
    /// If the `j`-th private hand is not locked, returns `-1.0` for all `i`.
    #[inline]
    pub fn current_locking_strategy(&self) -> Option<Vec<f32>> {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let offset = self.node_offset(self.node());
        self.locking_strategy.get(&offset).map(|s| {
            let mut ret = s.clone();
            let player = self.current_player();
            let num_hands = self.num_private_hands(player);
            ret.chunks_exact_mut(num_hands).for_each(|chunk| {
                self.apply_swap(chunk, player, false);
            });
            ret
        })
    }

    /// Returns the reference to the current node.
    #[inline]
    fn node(&self) -> &PostFlopNode {
        unsafe { &*self.node_ptr }
    }

    /// Returns the mutable reference to the current node.
    #[inline]
    fn node_mut(&mut self) -> &mut PostFlopNode {
        unsafe { &mut *self.node_ptr }
    }

    /// Returns the offset of the given node.
    #[inline]
    fn node_offset(&self, node: &PostFlopNode) -> usize {
        if self.is_compression_enabled {
            let base = self.storage1_compressed.lock().as_ptr();
            unsafe { (node.storage1 as *const u16).offset_from(base) as usize }
        } else {
            let base = self.storage1.lock().as_ptr();
            unsafe { (node.storage1 as *const f32).offset_from(base) as usize }
        }
    }

    /// Returns a card list of isomorphic chances.
    #[inline]
    fn isomorphic_cards(&self, node: &PostFlopNode) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism_card
        } else {
            &self.river_isomorphism_card[node.turn as usize]
        }
    }

    /// Applies the swap.
    #[inline]
    fn apply_swap(&self, slice: &mut [f32], player: usize, reverse: bool) {
        let turn_swap = self
            .turn_swap
            .map(|suit| &self.turn_isomorphism_swap[suit as usize][player]);

        let river_swap = self.river_swap.map(|(turn, suit)| {
            &self.river_isomorphism_swap[turn as usize & 3][suit as usize][player]
        });

        let swaps = if !reverse {
            [turn_swap, river_swap]
        } else {
            [river_swap, turn_swap]
        };

        for swap in swaps.into_iter().flatten() {
            for &(i, j) in swap {
                slice.swap(i as usize, j as usize);
            }
        }
    }

    /// Internal method for calculating the equity.
    fn equity_internal(&self, result: &mut [f64], player: usize, turn: u8, river: u8, amount: f64) {
        let pair_index = card_pair_index(turn, river);
        let hand_strength = &self.hand_strength[pair_index];
        let player_strength = &hand_strength[player];
        let opponent_strength = &hand_strength[player ^ 1];

        let player_len = player_strength.len();
        let opponent_len = opponent_strength.len();

        if player_len == 0 || opponent_len == 0 {
            return;
        }

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let opponent_weights = &self.weights[player ^ 1];
        let mut weight_sum = 0.0;
        let mut weight_minus = [0.0; 52];

        let valid_player_strength = &player_strength[1..player_len - 1];
        let mut i = 1;

        for &StrengthItem { strength, index } in valid_player_strength {
            unsafe {
                while opponent_strength.get_unchecked(i).strength < strength {
                    let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let weight_i = *opponent_weights.get_unchecked(opponent_index) as f64;
                    weight_sum += weight_i;
                    *weight_minus.get_unchecked_mut(c1 as usize) += weight_i;
                    *weight_minus.get_unchecked_mut(c2 as usize) += weight_i;
                    i += 1;
                }
                let (c1, c2) = *player_cards.get_unchecked(index as usize);
                let opponent_weight = weight_sum
                    - weight_minus.get_unchecked(c1 as usize)
                    - weight_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(index as usize) += amount * opponent_weight;
            }
        }

        weight_sum = 0.0;
        weight_minus.fill(0.0);
        i = opponent_len - 2;

        for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
            unsafe {
                while opponent_strength.get_unchecked(i).strength > strength {
                    let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let weight_i = *opponent_weights.get_unchecked(opponent_index) as f64;
                    weight_sum += weight_i;
                    *weight_minus.get_unchecked_mut(c1 as usize) += weight_i;
                    *weight_minus.get_unchecked_mut(c2 as usize) += weight_i;
                    i -= 1;
                }
                let (c1, c2) = *player_cards.get_unchecked(index as usize);
                let opponent_weight = weight_sum
                    - weight_minus.get_unchecked(c1 as usize)
                    - weight_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(index as usize) -= amount * opponent_weight;
            }
        }
    }

    /// Returns the counterfactual values vector of the chance node.
    fn cfvalues_chance(&self, player: usize) -> Vec<f32> {
        let num_hands = self.num_private_hands(player);
        let storage = self.node().cfvalue_storage(player);

        if storage == CfValueStorage::None {
            return self.cfvalues_cache[player].to_vec();
        }

        let mut vec = if self.is_compression_enabled {
            let slice = self.node().cfvalues_chance_compressed(player);
            let scale = self.node().cfvalue_chance_scale(player);
            decode_signed_slice(slice, scale)
        } else {
            self.node().cfvalues_chance(player).to_vec()
        };

        if storage == CfValueStorage::Sum {
            return vec;
        }

        let mut ret_f64 = vec![0.0; num_hands];

        vec.chunks_exact(num_hands).for_each(|row| {
            ret_f64.iter_mut().zip(row).for_each(|(r, &v)| {
                *r += v as f64;
            });
        });

        let isomorphic_chances = self.isomorphic_chances(self.node());

        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &self.isomorphic_swap(self.node(), i)[player];
            let tmp = row_mut(&mut vec, isomorphic_index as usize, num_hands);

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }

            ret_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }
        }

        ret_f64.iter().map(|&v| v as f32).collect()
    }
}

impl Default for PostFlopGame {
    #[inline]
    fn default() -> Self {
        Self {
            state: State::default(),
            card_config: CardConfig::default(),
            tree_config: TreeConfig::default(),
            added_lines: Vec::default(),
            removed_lines: Vec::default(),
            action_root: Box::default(),
            num_combinations: 0.0,
            initial_weights: Default::default(),
            private_cards: Default::default(),
            same_hand_index: Default::default(),
            valid_indices_flop: Default::default(),
            valid_indices_turn: Default::default(),
            valid_indices_river: Default::default(),
            hand_strength: Vec::default(),
            turn_isomorphism_ref: Vec::default(),
            turn_isomorphism_card: Vec::default(),
            turn_isomorphism_swap: Default::default(),
            river_isomorphism_ref: Vec::default(),
            river_isomorphism_card: Vec::default(),
            river_isomorphism_swap: Default::default(),
            is_compression_enabled: false,
            misc_memory_usage: 0,
            num_storage_actions: 0,
            num_storage_chances: 0,
            node_arena: Vec::default(),
            storage1: MutexLike::default(),
            storage2: MutexLike::default(),
            storage_chance: MutexLike::default(),
            storage1_compressed: MutexLike::default(),
            storage2_compressed: MutexLike::default(),
            storage_chance_compressed: MutexLike::default(),
            locking_strategy: BTreeMap::default(),
            root_cfvalue_ip: Vec::default(),
            history: Vec::default(),
            is_normalized_weight_cached: false,
            node_ptr: ptr::null_mut(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
            chance_factor: 0,
            turn_swapped_suit: None,
            turn_swap: None,
            river_swap: None,
            total_bet_amount: [0, 0],
            prev_bet_amount: 0,
            weights: Default::default(),
            normalized_weights: Default::default(),
            cfvalues_cache: Default::default(),
        }
    }
}

#[cfg(feature = "bincode")]
static VERSION_STR: &str = "2023-02-23";

#[cfg(feature = "bincode")]
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

#[cfg(feature = "bincode")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::*;
    use crate::solver::*;

    #[cfg(feature = "bincode")]
    use std::{
        fs::File,
        io::{BufReader, BufWriter, Write},
    };

    #[test]
    fn all_check_all_range() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);

        game.play(0);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);

        game.play(0);
        assert!(game.is_chance_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);

        game.play(usize::MAX);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);

        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(0);
        game.play(0);
        assert!(game.is_terminal_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 37.5).abs() < 1e-4);
        assert!((ev_ip - 22.5).abs() < 1e-4);

        game.play(0);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 37.5).abs() < 1e-4);
        assert!((ev_ip - 22.5).abs() < 1e-4);

        game.play(0);
        assert!(game.is_chance_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 37.5).abs() < 1e-4);
        assert!((ev_ip - 22.5).abs() < 1e-4);

        game.play(usize::MAX);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 37.5).abs() < 1e-4);
        assert!((ev_ip - 22.5).abs() < 1e-4);

        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(1);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 75.0).abs() < 1e-4);
        assert!((ev_ip - 15.0).abs() < 1e-4);

        game.play(1);
        assert!(game.is_terminal_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 60.0).abs() < 1e-4);
        assert!((ev_ip - 60.0).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range_compressed() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(true);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-4);
        assert!((equity_ip - 0.5).abs() < 1e-4);
        assert!((ev_oop - 37.5).abs() < 1e-2);
        assert!((ev_ip - 22.5).abs() < 1e-2);

        game.play(0);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-4);
        assert!((equity_ip - 0.5).abs() < 1e-4);
        assert!((ev_oop - 37.5).abs() < 1e-2);
        assert!((ev_ip - 22.5).abs() < 1e-2);

        game.play(0);
        assert!(game.is_chance_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-4);
        assert!((equity_ip - 0.5).abs() < 1e-4);
        assert!((ev_oop - 37.5).abs() < 1e-2);
        assert!((ev_ip - 22.5).abs() < 1e-2);

        game.play(usize::MAX);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-4);
        assert!((equity_ip - 0.5).abs() < 1e-4);
        assert!((ev_oop - 37.5).abs() < 1e-2);
        assert!((ev_ip - 22.5).abs() < 1e-2);

        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(1);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-4);
        assert!((equity_ip - 0.5).abs() < 1e-4);
        assert!((ev_oop - 75.0).abs() < 1e-2);
        assert!((ev_ip - 15.0).abs() < 1e-2);

        game.play(1);
        assert!(game.is_terminal_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-4);
        assert!((equity_ip - 0.5).abs() < 1e-4);
        assert!((ev_oop - 60.0).abs() < 1e-2);
        assert!((ev_ip - 60.0).abs() < 1e-2);
    }

    #[test]
    fn one_raise_all_range_with_turn() {
        let card_config = CardConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            range: [Range::ones(); 2],
            turn: card_from_str("Qc").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 60,
            effective_stack: 970,
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.5).abs() < 1e-5);
        assert!((root_equity_ip - 0.5).abs() < 1e-5);
        assert!((root_ev_oop - 37.5).abs() < 1e-4);
        assert!((root_ev_ip - 22.5).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range_with_river() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            turn: card_from_str("Qc").unwrap(),
            river: card_from_str("7s").unwrap(),
        };

        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 60,
            effective_stack: 970,
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 37.5).abs() < 1e-4);
        assert!((ev_ip - 22.5).abs() < 1e-4);

        game.play(0);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);

        game.play(0);
        assert!(game.is_terminal_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 30.0).abs() < 1e-4);
        assert!((ev_ip - 30.0).abs() < 1e-4);

        game.back_to_root();
        game.play(1);
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 75.0).abs() < 1e-4);
        assert!((ev_ip - 15.0).abs() < 1e-4);

        game.play(0);
        assert!(game.is_terminal_node());
        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!(game.is_terminal_node());
        assert!((equity_oop - 0.5).abs() < 1e-5);
        assert!((equity_ip - 0.5).abs() < 1e-5);
        assert!((ev_oop - 90.0).abs() < 1e-4);
        assert!((ev_ip - 0.0).abs() < 1e-4);
    }

    #[test]
    fn always_win() {
        // be careful for straight flushes
        let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let card_config = CardConfig {
            range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
            flop: flop_from_str("AcAdKh").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 1.0).abs() < 1e-5);
        assert!((equity_ip - 0.0).abs() < 1e-5);
        assert!((ev_oop - 60.0).abs() < 1e-4);
        assert!((ev_ip - 0.0).abs() < 1e-4);

        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(0);
        game.play(0);
        assert!(game.is_terminal_node());

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let equity_oop = compute_average(&game.equity(0), weights_oop);
        let equity_ip = compute_average(&game.equity(1), weights_ip);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((equity_oop - 1.0).abs() < 1e-5);
        assert!((equity_ip - 0.0).abs() < 1e-5);
        assert!((ev_oop - 60.0).abs() < 1e-4);
        assert!((ev_ip - 0.0).abs() < 1e-4);
    }

    #[test]
    fn always_win_raked() {
        // be careful for straight flushes
        let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let card_config = CardConfig {
            range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
            flop: flop_from_str("AcAdKh").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            rake_rate: 0.05,
            rake_cap: 10.0,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((ev_oop - 57.0).abs() < 1e-4);
        assert!((ev_ip - 0.0).abs() < 1e-4);

        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(0);
        game.play(0);
        assert!(game.is_chance_node());
        game.play(usize::MAX);
        game.play(0);
        game.play(0);
        assert!(game.is_terminal_node());

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let ev_ip = compute_average(&game.expected_values(1), weights_ip);
        assert!((ev_oop - 57.0).abs() < 1e-4);
        assert!((ev_ip - 0.0).abs() < 1e-4);
    }

    #[test]
    fn always_lose() {
        // be careful for straight flushes
        let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let card_config = CardConfig {
            range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
            flop: flop_from_str("AcAdKh").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.0).abs() < 1e-5);
        assert!((root_equity_ip - 1.0).abs() < 1e-5);
        assert!((root_ev_oop - 0.0).abs() < 1e-4);
        assert!((root_ev_ip - 60.0).abs() < 1e-4);
    }

    #[test]
    fn always_lose_raked() {
        // be careful for straight flushes
        let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let card_config = CardConfig {
            range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
            flop: flop_from_str("AcAdKh").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            rake_rate: 0.05,
            rake_cap: 10.0,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_ev_oop - 0.0).abs() < 1e-4);
        assert!((root_ev_ip - 57.0).abs() < 1e-4);
    }

    #[test]
    fn always_tie() {
        let card_config = CardConfig {
            range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
            flop: flop_from_str("2c6dTh").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.5).abs() < 1e-5);
        assert!((root_equity_ip - 0.5).abs() < 1e-5);
        assert!((root_ev_oop - 30.0).abs() < 1e-4);
        assert!((root_ev_ip - 30.0).abs() < 1e-4);
    }

    #[test]
    fn always_tie_raked() {
        let card_config = CardConfig {
            range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
            flop: flop_from_str("2c6dTh").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            rake_rate: 0.05,
            rake_cap: 10.0,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_ev_oop - 28.5).abs() < 1e-4);
        assert!((root_ev_ip - 28.5).abs() < 1e-4);
    }

    #[test]
    fn no_assignment() {
        let card_config = CardConfig {
            range: ["TT".parse().unwrap(), "TT".parse().unwrap()],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let game = PostFlopGame::with_config(card_config, action_tree);
        assert!(game.is_err());
    }

    // #[test]
    // fn remove_lines() {
    //     use crate::bet_size::BetSizeCandidates;
    //     let card_config = CardConfig {
    //         range: ["TT+,AKo,AQs+".parse().unwrap(), "AA".parse().unwrap()],
    //         flop: flop_from_str("2c6dTh").unwrap(),
    //         ..Default::default()
    //     };

    //     // Simple tree: force checks on flop, and only use 1/2 pot bets on turn and river
    //     let tree_config = TreeConfig {
    //         starting_pot: 60,
    //         effective_stack: 970,
    //         turn_bet_sizes: [
    //             BetSizeCandidates::try_from(("50%", "")).unwrap(),
    //             Default::default(),
    //         ],
    //         river_bet_sizes: [
    //             BetSizeCandidates::try_from(("50%", "")).unwrap(),
    //             Default::default(),
    //         ],
    //         ..Default::default()
    //     };

    //     let action_tree = ActionTree::new(tree_config).unwrap();
    //     let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    //     let lines = vec![
    //         vec![
    //             Action::Check,
    //             Action::Check,
    //             Action::Chance(2),
    //             Action::Bet(30),
    //         ],
    //         vec![
    //             Action::Check,
    //             Action::Check,
    //             Action::Chance(2),
    //             Action::Check,
    //             Action::Check,
    //             Action::Chance(3),
    //             Action::Bet(30),
    //         ],
    //     ];

    //     let res = game.remove_lines(&lines);
    //     assert!(res.is_ok());

    //     game.allocate_memory(false);

    //     // Check that the turn line is removed
    //     game.back_to_root();
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(2);
    //     assert_eq!(game.available_actions(), &[Action::Check]);
    //     assert_eq!(game.node().children.len(), 1);

    //     // Check that other turn lines are correct
    //     game.back_to_root();
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(3);
    //     assert_eq!(game.available_actions(), &[Action::Check, Action::Bet(30)]);
    //     assert_eq!(game.node().children.len(), 2);

    //     // Check that the river line is removed
    //     game.back_to_root();
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(2);
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(3);

    //     assert_eq!(game.available_actions(), &[Action::Check]);
    //     assert_eq!(game.node().children.len(), 1);

    //     // Check that other river lines are correct
    //     game.back_to_root();
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(2);
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(4);

    //     assert_eq!(game.available_actions(), &[Action::Check, Action::Bet(30)]);
    //     assert_eq!(game.node().children.len(), 2);

    //     game.back_to_root();
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(3);
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(game.node().actions.binary_search(&Action::Check).unwrap());
    //     game.play(4);

    //     assert_eq!(game.available_actions(), &[Action::Check, Action::Bet(30)]);
    //     assert_eq!(game.node().children.len(), 2);

    //     solve(&mut game, 10, 0.05, false);
    // }

    #[test]
    fn isomorphism_monotone() {
        let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
        let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

        let card_config = CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str("QhJh2h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        let mut check = |history: &[usize],
                         expected_turn_swap: Option<u8>,
                         expected_river_swap: Option<(u8, u8)>| {
            game.apply_history(history);
            game.cache_normalized_weights();
            let weights = game.normalized_weights(0);
            let ev = game.expected_values(0);
            weights.iter().zip(ev.iter()).for_each(|(&w, &v)| {
                assert!(!(w > 0.0 && v == 50.0));
            });
            assert_eq!(game.turn_swap, expected_turn_swap);
            assert_eq!(game.river_swap, expected_river_swap);
        };

        check(&[0, 0, 4], None, None);
        check(&[0, 0, 5], Some(1), None);
        check(&[0, 0, 6], None, None);
        check(&[0, 0, 7], Some(3), None);

        check(&[0, 0, 4, 0, 0, 8], None, None);
        check(&[0, 0, 4, 0, 0, 9], None, None);
        check(&[0, 0, 4, 0, 0, 10], None, None);
        check(&[0, 0, 4, 0, 0, 11], None, Some((4, 3)));

        check(&[0, 0, 5, 0, 0, 8], Some(1), None);
        check(&[0, 0, 5, 0, 0, 9], Some(1), None);
        check(&[0, 0, 5, 0, 0, 10], Some(1), None);
        check(&[0, 0, 5, 0, 0, 11], Some(1), Some((5, 3)));

        check(&[0, 0, 6, 0, 0, 8], None, None);
        check(&[0, 0, 6, 0, 0, 9], None, Some((6, 1)));
        check(&[0, 0, 6, 0, 0, 10], None, None);
        check(&[0, 0, 6, 0, 0, 11], None, Some((6, 3)));

        check(&[0, 0, 7, 0, 0, 8], Some(3), Some((7, 1)));
        check(&[0, 0, 7, 0, 0, 9], Some(3), None);
        check(&[0, 0, 7, 0, 0, 10], Some(3), None);
        check(&[0, 0, 7, 0, 0, 11], Some(3), None);
    }

    #[test]
    fn node_locking() {
        let card_config = CardConfig {
            range: ["AsAh,QsQh".parse().unwrap(), "KsKh".parse().unwrap()],
            flop: flop_from_str("2s3h4d").unwrap(),
            turn: card_from_str("6c").unwrap(),
            river: card_from_str("7c").unwrap(),
        };

        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 20,
            effective_stack: 10,
            river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        game.play(1); // all-in
        game.lock_current_strategy(&[0.25, 0.75]); // 25% fold, 75% call
        game.back_to_root();

        solve(&mut game, 1000, 0.0, false);
        game.cache_normalized_weights();

        let ev_oop = game.expected_values(0);
        let ev_ip = game.expected_values(1);
        assert!((ev_oop[0] - 0.0).abs() < 1e-2);
        assert!((ev_oop[1] - 27.5).abs() < 5e-2);
        assert!((ev_ip[0] - 6.25).abs() < 1e-2);

        let strategy_oop = game.strategy();
        assert!((strategy_oop[0] - 1.0).abs() < 1e-3); // QQ check
        assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
        assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // QQ bet
        assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet

        game.allocate_memory(false);
        game.play(1); // all-in
        game.lock_current_strategy(&[0.5, 0.5]); // 50% fold, 50% call
        game.back_to_root();

        solve(&mut game, 1000, 0.0, false);
        game.cache_normalized_weights();

        let ev_oop = game.expected_values(0);
        let ev_ip = game.expected_values(1);
        assert!((ev_oop[0] - 5.0).abs() < 1e-2);
        assert!((ev_oop[1] - 25.0).abs() < 5e-2);
        assert!((ev_ip[0] - 5.0).abs() < 1e-2);

        let strategy_oop = game.strategy();
        assert!((strategy_oop[0] - 0.0).abs() < 1e-3); // QQ check
        assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
        assert!((strategy_oop[2] - 1.0).abs() < 1e-3); // QQ bet
        assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet
    }

    #[test]
    fn node_locking_partial() {
        let card_config = CardConfig {
            range: ["AsAh,QsQh,JsJh".parse().unwrap(), "KsKh".parse().unwrap()],
            flop: flop_from_str("2s3h4d").unwrap(),
            turn: card_from_str("6c").unwrap(),
            river: card_from_str("7c").unwrap(),
        };

        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 10,
            effective_stack: 10,
            river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        game.lock_current_strategy(&[0.8, 0.0, 0.0, 0.2, 0.0, 0.0]); // JJ -> 80% check, 20% all-in

        solve(&mut game, 1000, 0.0, false);
        game.cache_normalized_weights();

        let ev_oop = game.expected_values(0);
        let ev_ip = game.expected_values(1);
        assert!((ev_oop[0] - 0.0).abs() < 1e-2);
        assert!((ev_oop[1] - 0.0).abs() < 1e-2);
        assert!((ev_oop[2] - 15.0).abs() < 5e-2);
        assert!((ev_ip[0] - 5.0).abs() < 1e-2);

        let strategy_oop = game.strategy();
        assert!((strategy_oop[0] - 0.8).abs() < 1e-3); // JJ check
        assert!((strategy_oop[1] - 0.7).abs() < 1e-3); // QQ check
        assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // AA check
        assert!((strategy_oop[3] - 0.2).abs() < 1e-3); // JJ bet
        assert!((strategy_oop[4] - 0.3).abs() < 1e-3); // QQ bet
        assert!((strategy_oop[5] - 1.0).abs() < 1e-3); // AA bet
    }

    #[test]
    fn node_locking_isomorphism() {
        let card_config = CardConfig {
            range: ["AKs".parse().unwrap(), "AKs".parse().unwrap()],
            flop: flop_from_str("2c3c4c").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 10,
            effective_stack: 10,
            river_bet_sizes: [("a", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        game.apply_history(&[0, 0, 15, 0, 0, 14]); // Turn: Spades, River: Hearts
        game.lock_current_strategy(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // AhKh -> check

        finalize(&mut game);

        game.apply_history(&[0, 0, 13, 0, 0, 14]);
        assert_eq!(
            game.strategy(),
            vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
        );

        game.apply_history(&[0, 0, 13, 0, 0, 15]);
        assert_eq!(
            game.strategy(),
            vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
        );

        game.apply_history(&[0, 0, 14, 0, 0, 13]);
        assert_eq!(
            game.strategy(),
            vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
        );

        game.apply_history(&[0, 0, 14, 0, 0, 15]);
        assert_eq!(
            game.strategy(),
            vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
        );

        game.apply_history(&[0, 0, 15, 0, 0, 13]);
        assert_eq!(
            game.strategy(),
            vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
        );

        game.apply_history(&[0, 0, 15, 0, 0, 14]);
        assert_eq!(
            game.strategy(),
            vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
        );
    }

    #[test]
    #[cfg(feature = "bincode")]
    fn serialize_and_deserialize() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        let config = bincode::config::legacy();

        // save
        let file = File::create("tmpfile.bin").unwrap();
        let mut write_buf = BufWriter::new(file);
        bincode::encode_into_std_write(&game, &mut write_buf, config).unwrap();
        write_buf.flush().unwrap();

        // load
        let file = File::open("tmpfile.bin").unwrap();
        let mut read_buf = BufReader::new(file);
        let mut game: PostFlopGame = bincode::decode_from_std_read(&mut read_buf, config).unwrap();

        // remove tmpfile
        std::fs::remove_file("tmpfile.bin").unwrap();

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.5).abs() < 1e-5);
        assert!((root_equity_ip - 0.5).abs() < 1e-5);
        assert!((root_ev_oop - 37.5).abs() < 1e-4);
        assert!((root_ev_ip - 22.5).abs() < 1e-4);
    }

    #[test]
    #[ignore]
    fn solve_pio_preset_normal() {
        let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
        let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

        let card_config = CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str("QsJh2h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 180,
            effective_stack: 910,
            flop_bet_sizes: [
                ("52%", "45%").try_into().unwrap(),
                ("52%", "45%").try_into().unwrap(),
            ],
            turn_bet_sizes: [
                ("55%", "45%").try_into().unwrap(),
                ("55%", "45%").try_into().unwrap(),
            ],
            river_bet_sizes: [
                ("70%", "45%").try_into().unwrap(),
                ("70%", "45%").try_into().unwrap(),
            ],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        println!(
            "memory usage: {:.2}GB",
            game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        game.allocate_memory(false);

        solve(&mut game, 1000, 180.0 * 0.001, true);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        // verified by PioSOLVER Free
        assert!((root_equity_oop - 0.55347).abs() < 1e-5);
        assert!((root_equity_ip - 0.44653).abs() < 1e-5);
        assert!((root_ev_oop - 105.11).abs() < 0.2);
        assert!((root_ev_ip - 74.89).abs() < 0.2);
    }

    #[test]
    #[ignore]
    fn solve_pio_preset_raked() {
        let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
        let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

        let card_config = CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str("QsJh2h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 180,
            effective_stack: 910,
            rake_rate: 0.05,
            rake_cap: 30.0,
            flop_bet_sizes: [
                ("52%", "45%").try_into().unwrap(),
                ("52%", "45%").try_into().unwrap(),
            ],
            turn_bet_sizes: [
                ("55%", "45%").try_into().unwrap(),
                ("55%", "45%").try_into().unwrap(),
            ],
            river_bet_sizes: [
                ("70%", "45%").try_into().unwrap(),
                ("70%", "45%").try_into().unwrap(),
            ],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        println!(
            "memory usage: {:.2}GB",
            game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        game.allocate_memory(false);

        solve(&mut game, 1000, 180.0 * 0.001, true);

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        // verified by PioSOLVER Free (but not theoretically guaranteed to be the same)
        assert!((root_ev_oop - 95.57).abs() < 0.2);
        assert!((root_ev_ip - 66.98).abs() < 0.2);
    }
}
