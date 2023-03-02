mod evaluation;
mod interpreter;
mod node;

#[cfg(feature = "bincode")]
mod serialization;

#[cfg(test)]
mod tests;

use crate::action_tree::*;
use crate::card::*;
use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use node::*;
use std::collections::BTreeMap;
use std::mem::{self, size_of, MaybeUninit};
use std::ptr;

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

#[derive(Default)]
struct BuildTreeInfo {
    flop_index: usize,
    turn_index: usize,
    river_index: usize,
    num_storage: u64,
    num_storage_ip: u64,
    num_storage_chance: u64,
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

    #[inline]
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        self.evaluate_internal(result, node, player, cfreach);
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.state == State::Solved
    }

    #[inline]
    fn set_solved(&mut self) {
        self.state = State::Solved;
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
            let index = self.node_index(node);
            self.locking_strategy.get(&index).unwrap()
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

        let num_elements = 2 * self.num_storage + self.num_storage_ip + self.num_storage_chance;
        let uncompressed = 4 * num_elements + self.misc_memory_usage;
        let compressed = 2 * num_elements + self.misc_memory_usage;

        (uncompressed, compressed)
    }

    /// Remove lines after building the `PostFlopGame` but before allocating memory.
    ///
    /// This allows the removal of chance-specific lines (e.g., remove overbets on board-pairing
    /// turns) which we cannot do while building an action tree.
    pub fn remove_lines(&mut self, lines: &[Vec<Action>]) -> Result<(), String> {
        if self.state <= State::Uninitialized {
            return Err("Game is not successfully initialized".to_string());
        } else if self.state >= State::MemoryAllocated {
            return Err("Game has already been allocated".to_string());
        }

        for line in lines {
            let mut root = self.root();
            let info = self.remove_line_recursive(&mut root, line)?;
            self.num_storage -= info.num_storage;
            self.num_storage_ip -= info.num_storage_ip;
            self.num_storage_chance -= info.num_storage_chance;
        }

        Ok(())
    }

    /// Allocates the memory.
    pub fn allocate_memory(&mut self, enable_compression: bool) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.state == State::MemoryAllocated && self.is_compression_enabled == enable_compression
        {
            return;
        }

        let num_bytes = if enable_compression { 2 } else { 4 };
        if num_bytes * self.num_storage > isize::MAX as u64
            || num_bytes * self.num_storage_chance > isize::MAX as u64
        {
            panic!("Memory usage exceeds maximum size");
        }

        self.state = State::MemoryAllocated;
        self.is_compression_enabled = enable_compression;

        self.clear_storage();

        let storage_bytes = (num_bytes * self.num_storage) as usize;
        let storage_ip_bytes = (num_bytes * self.num_storage_ip) as usize;
        let storage_chance_bytes = (num_bytes * self.num_storage_chance) as usize;

        self.storage1 = vec![0; storage_bytes];
        self.storage2 = vec![0; storage_bytes];
        self.storage_ip = vec![0; storage_ip_bytes];
        self.storage_chance = vec![0; storage_chance_bytes];

        self.allocate_memory_nodes();
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
            turn_index: num_nodes[0] as usize,
            river_index: (num_nodes[0] + num_nodes[1]) as usize,
            ..Default::default()
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

        self.num_storage = info.num_storage;
        self.num_storage_ip = info.num_storage_ip;
        self.num_storage_chance = info.num_storage_chance;
        self.misc_memory_usage = self.memory_usage_internal();

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

        self.weights = vecs.clone();
        self.normalized_weights = vecs.clone();
        self.cfvalues_cache = vecs;

        self.back_to_root();
    }

    /// Clears the storage.
    #[inline]
    fn clear_storage(&mut self) {
        self.storage1.clear();
        self.storage2.clear();
        self.storage_ip.clear();
        self.storage_chance.clear();
        self.storage1.shrink_to_fit();
        self.storage2.shrink_to_fit();
        self.storage_ip.shrink_to_fit();
        self.storage_chance.shrink_to_fit();
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

        node.num_elements = node
            .cfvalue_storage_player()
            .map_or(0, |player| self.num_private_hands(player)) as u32;

        info.num_storage_chance += node.num_elements as u64;
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
        node.num_children = action_node.children.len() as u16;
        *base += node.num_children as usize;

        for (child, action) in node.children().iter().zip(action_node.actions.iter()) {
            let mut child = child.lock();
            child.prev_action = *action;
            child.turn = node.turn;
            child.river = node.river;
        }

        let num_private_hands = self.num_private_hands(node.player as usize);
        node.num_elements = (node.num_actions() * num_private_hands) as u32;
        node.num_elements_ip = match node.prev_action {
            Action::None | Action::Chance(_) => self.num_private_hands(PLAYER_IP as usize) as u16,
            _ => 0,
        };

        info.num_storage += node.num_elements as u64;
        info.num_storage_ip += node.num_elements_ip as u64;
    }

    /// Calculates the number of storage elements that will be removed.
    fn calculate_removed_line_info_recursive(node: &mut PostFlopNode, info: &mut BuildTreeInfo) {
        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            info.num_storage_chance += node.num_elements as u64;
            node.num_elements = 0;
        } else {
            info.num_storage += node.num_elements as u64;
            info.num_storage_ip += node.num_elements_ip as u64;
            node.num_elements = 0;
            node.num_elements_ip = 0;
        }

        for action in node.action_indices() {
            Self::calculate_removed_line_info_recursive(&mut node.play(action), info);
        }
    }

    /// Remove a line from a `PostFlopGame` tree.
    fn remove_line_recursive(
        &self,
        node: &mut PostFlopNode,
        line: &[Action],
    ) -> Result<BuildTreeInfo, String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        let action = line[0];
        let search_result = node
            .children()
            .binary_search_by(|child| child.lock().prev_action.cmp(&action));

        if search_result.is_err() {
            return Err(format!("Action does not exist: {action:?}"));
        }

        let index = search_result.unwrap();
        if line.len() > 1 {
            let result = self.remove_line_recursive(&mut node.children()[index].lock(), &line[1..]);
            return result;
        }

        if node.is_chance() {
            return Err("Cannot remove a line ending in a chance action".to_string());
        }

        if node.num_actions() <= 1 {
            return Err("Cannot remove the last action from a node".to_string());
        }

        // Remove action/children at index. To do this we must
        // 1. compute the storage space required by the tree rooted at action index
        // 2. remove children and actions
        // 3. re-define `num_elements` after we remove children and actions

        // STEP 1
        let mut info = BuildTreeInfo {
            num_storage: self.num_private_hands(node.player as usize) as u64,
            ..Default::default()
        };

        let mut node_to_remove = node.play(index);
        Self::calculate_removed_line_info_recursive(&mut node_to_remove, &mut info);

        // STEP 2
        let children = node.children();
        for i in index..node.num_children as usize - 1 {
            let mut x = children[i].lock();
            let mut y = children[i + 1].lock();
            mem::swap(&mut *x, &mut *y);
            if x.children_offset > 0 {
                x.children_offset += 1;
            }
        }
        node.num_children -= 1;

        // STEP 3
        node.num_elements -= self.num_private_hands(node.player as usize) as u32;

        Ok(info)
    }

    /// Allocates memory recursively.
    fn allocate_memory_nodes(&mut self) {
        let num_bytes = if self.is_compression_enabled { 2 } else { 4 };
        let mut action_counter = 0;
        let mut ip_counter = 0;
        let mut chance_counter = 0;

        for node in &self.node_arena {
            let mut node = node.lock();
            if node.is_terminal() {
                // do nothing
            } else if node.is_chance() {
                unsafe {
                    let ptr = self.storage_chance.as_mut_ptr();
                    node.storage1 = ptr.add(chance_counter);
                }
                chance_counter += num_bytes * node.num_elements as usize;
            } else {
                unsafe {
                    let ptr1 = self.storage1.as_mut_ptr();
                    let ptr2 = self.storage2.as_mut_ptr();
                    let ptr3 = self.storage_ip.as_mut_ptr();
                    node.storage1 = ptr1.add(action_counter);
                    node.storage2 = ptr2.add(action_counter);
                    node.storage3 = ptr3.add(ip_counter);
                }
                action_counter += num_bytes * node.num_elements as usize;
                ip_counter += num_bytes * node.num_elements_ip as usize;
            }
        }
    }
}
