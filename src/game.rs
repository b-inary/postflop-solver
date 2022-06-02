use crate::bet_size::*;
use crate::interface::*;
use crate::mutex_like::*;
use crate::range::*;
use holdem_hand_evaluator::Hand;
use std::cmp;
use std::mem;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "custom_alloc")]
use crate::alloc::*;

/// A struct representing a post-flop game.
#[derive(Default)]
pub struct PostFlopGame {
    // Post-flop game configuration.
    config: GameConfig,

    // computed from `config`
    root: MutexLike<PostFlopNode>,
    num_combinations_inv: f64,
    initial_weight: [Vec<f32>; 2],
    private_hand_cards: [Vec<(u8, u8)>; 2],
    same_hand_index: [Vec<Option<usize>>; 2],
    hand_strength: Vec<[Vec<(usize, usize)>; 2]>,
    turn_isomorphism: Vec<usize>,
    turn_isomorphism_card: Vec<u8>,
    turn_isomorphism_swap: [[Vec<(usize, usize)>; 2]; 4],
    river_isomorphism: Vec<Vec<usize>>,
    river_isomorphism_card: Vec<Vec<u8>>,
    river_isomorphism_swap: Vec<[[Vec<(usize, usize)>; 2]; 4]>,
    is_memory_allocated: bool,
    is_compression_enabled: bool,
    num_storage_elements: u64,
    memory_usage: u64,
    memory_usage_compressed: u64,

    // global storage
    cum_regret: MutexLike<Vec<f32>>,
    strategy: MutexLike<Vec<f32>>,
    cum_regret_compressed: MutexLike<Vec<i16>>,
    strategy_compressed: MutexLike<Vec<u16>>,
}

/// A struct representing a node in post-flop game tree.
pub struct PostFlopNode {
    player: u16,
    turn: u8,
    river: u8,
    amount: i32,
    children: Vec<(Action, MutexLike<PostFlopNode>)>,
    cum_regret: *mut f32,
    strategy: *mut f32,
    cum_regret_compressed: *mut i16,
    strategy_compressed: *mut u16,
    cum_regret_scale: f32,
    strategy_scale: f32,
    num_elements: usize,
    is_strategy_locked: bool,
}

unsafe impl Send for PostFlopNode {}
unsafe impl Sync for PostFlopNode {}

/// A struct for post-flop game configuration.
///
/// # Examples
/// ```
/// use postflop_solver::*;
///
/// let bet_sizes = bet_sizes_from_str("50%", "50%").unwrap();
///
/// let config = GameConfig {
///     flop: flop_from_str("Td9d6h").unwrap(),
///     starting_pot: 60,
///     effective_stack: 970,
///     range: [Range::ones(), Range::ones()],
///     flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GameConfig {
    /// Flop cards: each card must be unique and in range [0, 51].
    pub flop: [u8; 3],

    /// Initial pot size: must be greater than 0.
    pub starting_pot: i32,

    /// Initial effective stack size: must be greater than 0.
    pub effective_stack: i32,

    /// Initial range of each player.
    pub range: [Range; 2],

    /// Bet size candidates of each player in flop.
    pub flop_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player in turn.
    pub turn_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player in river.
    pub river_bet_sizes: [BetSizeCandidates; 2],

    /// Add all-in action when SPR is below this value (set 0 to disable).
    pub add_all_in_threshold: f32,

    /// Replace bet action with all-in action when the ratio of opponent's next bet size to the pot
    /// size will be less than this value (set 0 to disable).
    pub replace_all_in_threshold: f32,

    /// Enable bet size adjustment of last two bets.
    pub adjust_last_two_bet_sizes: bool,
}

/// Possible actions in a post-flop game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    None,
    Fold,
    Check,
    Call,
    Bet(i32),
    Raise(i32),
    AllIn(i32),
    Chance(u8),
}

#[derive(Debug, Clone)]
struct BuildTreeInfo<'a> {
    last_action: Action,
    last_bet: [i32; 2],
    allin_flag: bool,
    current_memory_usage: &'a AtomicU64,
    num_storage_elements: &'a AtomicU64,
    stack_size: [usize; 2],
    max_stack_size: &'a [AtomicUsize; 2],
}

/// The index of player who is out of position.
const PLAYER_OOP: u16 = 0;

/// The index of player who is in position.
const PLAYER_IP: u16 = 1;

const PLAYER_CHANCE: u16 = 0xff;
const PLAYER_MASK: u16 = 0xff;
const PLAYER_TERMINAL_FLAG: u16 = 0x100;
const PLAYER_FOLD_FLAG: u16 = 0x300;

const NOT_DEALT: u8 = 0xff;

#[cfg(not(feature = "custom_alloc"))]
#[inline]
fn align_up(size: usize) -> usize {
    size
}

#[inline]
fn atomic_set_max(atomic: &AtomicUsize, value: usize) {
    let mut v = atomic.load(Ordering::Relaxed);
    while v < value {
        match atomic.compare_exchange_weak(v, value, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return,
            Err(new_v) => v = new_v,
        }
    }
}

impl Game for PostFlopGame {
    type Node = PostFlopNode;

    #[inline]
    fn root(&self) -> MutexGuardLike<Self::Node> {
        self.root.lock()
    }

    #[inline]
    fn num_private_hands(&self, player: usize) -> usize {
        self.private_hand_cards[player].len()
    }

    #[inline]
    fn initial_weight(&self, player: usize) -> &[f32] {
        &self.initial_weight[player]
    }

    #[inline]
    fn isomorphic_chances(&self, node: &Self::Node) -> &[usize] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism
        } else {
            &self.river_isomorphism[node.turn as usize]
        }
    }

    #[inline]
    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(usize, usize)>; 2] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism_swap[self.turn_isomorphism_card[index] as usize & 3]
        } else {
            &self.river_isomorphism_swap[node.turn as usize]
                [self.river_isomorphism_card[node.turn as usize][index] as usize & 3]
        }
    }

    fn evaluate(&self, result: &mut [f32], node: &Self::Node, player: usize, cfreach: &[f32]) {
        let amount = self.config.starting_pot as f64 * 0.5 + node.amount as f64;
        let amount_normalized = amount * self.num_combinations_inv;

        let player_cards = &self.private_hand_cards[player];
        let opponent_cards = &self.private_hand_cards[player ^ 1];

        let mut cfreach_sum = 0.0;
        let mut cfreach_minus = [0.0; 52];

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let mut board_mask = 0u64;
            if node.turn != NOT_DEALT {
                board_mask |= 1 << node.turn;
            }
            if node.river != NOT_DEALT {
                board_mask |= 1 << node.river;
            }

            let folded_player = node.player & PLAYER_MASK;
            let payoff_normalized = if folded_player as usize == player {
                -amount_normalized
            } else {
                amount_normalized
            };

            for i in 0..cfreach.len() {
                unsafe {
                    let (c1, c2) = *opponent_cards.get_unchecked(i);
                    let hand_mask: u64 = (1 << c1) | (1 << c2);
                    if hand_mask & board_mask == 0 {
                        let cfreach_i = *cfreach.get_unchecked(i) as f64;
                        cfreach_sum += cfreach_i;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i;
                    }
                }
            }

            let same_hand_index = &self.same_hand_index[player];
            for i in 0..result.len() {
                unsafe {
                    let (c1, c2) = *player_cards.get_unchecked(i);
                    let hand_mask: u64 = (1 << c1) | (1 << c2);
                    if hand_mask & board_mask == 0 {
                        // inclusion-exclusion principle
                        let cfreach = cfreach_sum
                            - *cfreach_minus.get_unchecked(c1 as usize)
                            - *cfreach_minus.get_unchecked(c2 as usize)
                            + same_hand_index
                                .get_unchecked(i)
                                .map_or(0.0, |j| *cfreach.get_unchecked(j) as f64);
                        *result.get_unchecked_mut(i) = (payoff_normalized * cfreach) as f32;
                    }
                }
            }
        }
        // showdown
        else {
            let hand_strength = &self.hand_strength[card_pair_index(node.turn, node.river)];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let mut j = 0;
            let player_len = player_strength.len();
            let opponent_len = opponent_strength.len();

            for i in 0..player_len {
                unsafe {
                    let (val, index) = *player_strength.get_unchecked(i);
                    while j < opponent_len && opponent_strength.get_unchecked(j).0 < val {
                        let opponent_index = opponent_strength.get_unchecked(j).1;
                        let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                        let cfreach_opp = *cfreach.get_unchecked(opponent_index) as f64;
                        cfreach_sum += cfreach_opp;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_opp;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_opp;
                        j += 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index) = (amount_normalized * cfreach) as f32;
                }
            }

            cfreach_sum = 0.0;
            cfreach_minus.fill(0.0);
            j = opponent_len;

            for i in (0..player_len).rev() {
                unsafe {
                    let (val, index) = *player_strength.get_unchecked(i);
                    while j > 0 && opponent_strength.get_unchecked(j - 1).0 > val {
                        let opponent_index = opponent_strength.get_unchecked(j - 1).1;
                        let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                        let cfreach_opp = *cfreach.get_unchecked(opponent_index) as f64;
                        cfreach_sum += cfreach_opp;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_opp;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_opp;
                        j -= 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index) -= (amount_normalized * cfreach) as f32;
                }
            }
        }
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.is_memory_allocated
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }
}

impl PostFlopGame {
    /// Constructs a new empty [`PostFlopGame`] (needs `update_config`).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs a new [`PostFlopGame`] instance with the given configuration.
    #[inline]
    pub fn with_config(config: &GameConfig) -> Result<Self, String> {
        let mut game = Self::default();
        game.update_config(config)?;
        Ok(game)
    }

    /// Updates the game configuration.
    #[inline]
    pub fn update_config(&mut self, config: &GameConfig) -> Result<(), String> {
        self.config = config.clone();
        self.check_config()?;
        self.init();
        Ok(())
    }

    /// Obtains the game configuration.
    #[inline]
    pub fn config(&self) -> &GameConfig {
        &self.config
    }

    /// Returns the card list of private hands of the given player.
    #[inline]
    pub fn private_hand_cards(&self, player: usize) -> &[(u8, u8)] {
        &self.private_hand_cards[player]
    }

    /// Returns the estimated memory usage in bytes (uncompressed, compressed).
    #[inline]
    pub fn memory_usage(&self) -> (u64, u64) {
        (self.memory_usage, self.memory_usage_compressed)
    }

    /// Allocates the memory.
    pub fn allocate_memory(&mut self, enable_compression: bool) {
        if self.is_memory_allocated && self.is_compression_enabled == enable_compression {
            return;
        }

        self.clear_storage();

        self.is_memory_allocated = true;
        self.is_compression_enabled = enable_compression;

        let num_elems = self.num_storage_elements as usize;
        if enable_compression {
            self.cum_regret_compressed = MutexLike::new(vec![0; num_elems]);
            self.strategy_compressed = MutexLike::new(vec![0; num_elems]);
        } else {
            self.cum_regret = MutexLike::new(vec![0.0; num_elems]);
            self.strategy = MutexLike::new(vec![0.0; num_elems]);
        }

        let counter = AtomicUsize::new(0);
        self.allocate_memory_recursive(&mut self.root(), &counter);
    }

    /// Returns card list of isomorphic chances.
    #[inline]
    pub fn isomorphic_card(&self, node: &PostFlopNode) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism_card
        } else {
            &self.river_isomorphism_card[node.turn as usize]
        }
    }

    /// Checks the configuration for errors.
    fn check_config(&mut self) -> Result<(), String> {
        let flop = self.config.flop;

        if flop.iter().any(|&c| c == NOT_DEALT) {
            return Err("Flop cards not initialized".to_string());
        }

        if flop.iter().any(|&c| 52 <= c) {
            return Err(format!("Flop cards must be in [0, 52): flop = {:?}", flop));
        }

        if flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2] {
            return Err(format!("Flop cards must be unique: flop = {:?}", flop));
        }

        if self.config.starting_pot <= 0 {
            return Err(format!(
                "Initial pot must be positive: starting_pot = {}",
                self.config.starting_pot
            ));
        }

        if self.config.effective_stack <= 0 {
            return Err(format!(
                "Initial stack must be positive: effective_stack = {}",
                self.config.effective_stack
            ));
        }

        if self.config.range[PLAYER_OOP as usize].is_empty() {
            return Err("OOP range is empty".to_string());
        }

        if self.config.range[PLAYER_IP as usize].is_empty() {
            return Err("IP range is empty".to_string());
        }

        let mut num_combinations = 0.0;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        for c1 in 0..52 {
            for c2 in c1 + 1..52 {
                let oop_mask: u64 = (1 << c1) | (1 << c2);
                let oop_weight = self.config.range[0].get_weight_by_cards(c1, c2);
                if oop_mask & flop_mask == 0 && oop_weight > 0.0 {
                    for c3 in 0..52 {
                        for c4 in c3 + 1..52 {
                            let ip_mask: u64 = (1 << c3) | (1 << c4);
                            let ip_weight = self.config.range[1].get_weight_by_cards(c3, c4);
                            if ip_mask & (flop_mask | oop_mask) == 0 {
                                num_combinations += oop_weight as f64 * ip_weight as f64;
                            }
                        }
                    }
                }
            }
        }

        if num_combinations == 0.0 {
            return Err("Valid card assignment does not exist".to_string());
        }

        self.num_combinations_inv = 1.0 / num_combinations;

        Ok(())
    }

    /// Initializes the game.
    #[inline]
    fn init(&mut self) {
        self.init_range();
        self.init_isomorphism();
        self.init_hand_strength();
        self.init_root();
    }

    /// Initializes fields `initial_weight`, `private_hand_cards` and `same_hand_index`.
    fn init_range(&mut self) {
        let flop = self.config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        let range = &self.config.range;

        for player in 0..2 {
            let range = range[player];
            let initial_weight = &mut self.initial_weight[player];
            let private_hand_cards = &mut self.private_hand_cards[player];
            initial_weight.clear();
            private_hand_cards.clear();

            for card1 in 0..52 {
                for card2 in card1 + 1..52 {
                    let hand_mask: u64 = (1 << card1) | (1 << card2);
                    let weight = range.get_weight_by_cards(card1, card2);
                    if weight > 0.0 && hand_mask & flop_mask == 0 {
                        initial_weight.push(weight);
                        private_hand_cards.push((card1, card2));
                    }
                }
            }
        }

        for player in 0..2 {
            let same_hand_index = &mut self.same_hand_index[player];
            same_hand_index.clear();

            let player_hands = &self.private_hand_cards[player];
            let opponent_hands = &self.private_hand_cards[player ^ 1];
            for hand in player_hands {
                same_hand_index.push(opponent_hands.binary_search(hand).ok());
            }
        }
    }

    /// Initializes a field related to isomorphism.
    fn init_isomorphism(&mut self) {
        let range = &self.config.range;
        let mut suit_isomorphism = [0; 4];
        let mut next_index = 1;
        'outer: for suit2 in 1..4 {
            for suit1 in 0..suit2 {
                if range[PLAYER_OOP as usize].is_suit_isomorphic(suit1, suit2)
                    && range[PLAYER_IP as usize].is_suit_isomorphic(suit1, suit2)
                {
                    suit_isomorphism[suit2 as usize] = suit_isomorphism[suit1 as usize];
                    continue 'outer;
                }
            }
            suit_isomorphism[suit2 as usize] = next_index;
            next_index += 1;
        }

        let flop = self.config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let mut flop_rankset = [0; 4];
        for card in flop {
            let rank = card >> 2;
            let suit = card & 3;
            flop_rankset[suit as usize] |= 1 << rank;
        }

        self.turn_isomorphism.clear();
        self.turn_isomorphism_card.clear();
        self.turn_isomorphism_swap.iter_mut().for_each(|x| {
            x[0].clear();
            x[1].clear();
        });

        let mut isomorphic_suit = [None; 4];
        let mut reverse_table = [usize::MAX; 52 * 51 / 2];

        // turn isomorphism
        for suit1 in 1..4 {
            for suit2 in 0..suit1 {
                if flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                    && suit_isomorphism[suit1 as usize] == suit_isomorphism[suit2 as usize]
                {
                    isomorphic_suit[suit1 as usize] = Some(suit2);

                    let replacer = |card: u8| {
                        if card & 3 == suit1 {
                            card - suit1 + suit2
                        } else if card & 3 == suit2 {
                            card + suit1 - suit2
                        } else {
                            card
                        }
                    };

                    let swap_list = &mut self.turn_isomorphism_swap[suit1 as usize];

                    for player in 0..2 {
                        reverse_table.fill(usize::MAX);
                        let cards = &self.private_hand_cards[player];
                        for i in 0..cards.len() {
                            reverse_table[card_pair_index(cards[i].0, cards[i].1)] = i;
                        }

                        for i in 0..cards.len() {
                            let c1 = replacer(cards[i].0);
                            let c2 = replacer(cards[i].1);
                            let index = reverse_table[card_pair_index(c1, c2)];
                            if index != usize::MAX && i < index {
                                swap_list[player].push((i, index));
                            }
                        }
                    }

                    break;
                }
            }
        }

        let mut counter = 0;
        let mut indices = [0; 52];

        for card in 0..52 {
            if (1 << card) & flop_mask != 0 {
                continue;
            }

            let suit = card & 3;

            if let Some(replace_suit) = isomorphic_suit[suit as usize] {
                let replace_card = card - suit + replace_suit;
                self.turn_isomorphism.push(indices[replace_card as usize]);
                self.turn_isomorphism_card.push(card);
            } else {
                indices[card as usize] = counter;
                counter += 1;
            }
        }

        self.river_isomorphism.clear();
        self.river_isomorphism_card.clear();
        self.river_isomorphism_swap.clear();

        // river isomorphism
        for turn in 0..52 {
            self.river_isomorphism.push(Vec::new());
            self.river_isomorphism_card.push(Vec::new());
            self.river_isomorphism_swap.push(Default::default());

            if (1 << turn) & flop_mask != 0 {
                continue;
            }

            let river_isomorphism = self.river_isomorphism.last_mut().unwrap();
            let river_isomorphism_card = self.river_isomorphism_card.last_mut().unwrap();
            let river_isomorphism_swap = self.river_isomorphism_swap.last_mut().unwrap();

            let turn_mask = flop_mask | (1 << turn);
            let mut turn_rankset = flop_rankset;
            turn_rankset[turn as usize & 3] |= 1 << (turn >> 2);

            isomorphic_suit.fill(None);

            for suit1 in 1..4 {
                for suit2 in 0..suit1 {
                    if flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                        && turn_rankset[suit1 as usize] == turn_rankset[suit2 as usize]
                        && suit_isomorphism[suit1 as usize] == suit_isomorphism[suit2 as usize]
                    {
                        isomorphic_suit[suit1 as usize] = Some(suit2 as u8);

                        let replacer = |card: u8| {
                            if card & 3 == suit1 {
                                card - suit1 + suit2
                            } else if card & 3 == suit2 {
                                card + suit1 - suit2
                            } else {
                                card
                            }
                        };

                        let swap_list = &mut river_isomorphism_swap[suit1 as usize];

                        for player in 0..2 {
                            reverse_table.fill(usize::MAX);
                            let cards = &self.private_hand_cards[player];
                            for i in 0..cards.len() {
                                reverse_table[card_pair_index(cards[i].0, cards[i].1)] = i;
                            }

                            for i in 0..cards.len() {
                                let c1 = replacer(cards[i].0);
                                let c2 = replacer(cards[i].1);
                                let index = reverse_table[card_pair_index(c1, c2)];
                                if index != usize::MAX && i < index {
                                    swap_list[player].push((i, index));
                                }
                            }
                        }

                        break;
                    }
                }
            }

            counter = 0;
            indices.fill(0);

            for card in 0..52 {
                if (1 << card) & turn_mask != 0 {
                    continue;
                }

                let suit = card & 3;

                if let Some(replace_suit) = isomorphic_suit[suit as usize] {
                    let replace_card = card - suit + replace_suit;
                    river_isomorphism.push(indices[replace_card as usize]);
                    river_isomorphism_card.push(card);
                } else {
                    indices[card as usize] = counter;
                    counter += 1;
                }
            }
        }
    }

    /// Initializes a field `hand_strength`.
    fn init_hand_strength(&mut self) {
        let mut flop = Hand::new();
        for card in &self.config.flop {
            flop = flop.add_card(*card as usize);
        }

        self.hand_strength = vec![Default::default(); 52 * 51 / 2];
        let private_hand_cards = &self.private_hand_cards;

        for board1 in 0..52 {
            for board2 in board1 + 1..52 {
                if !flop.contains(board1 as usize) && !flop.contains(board2 as usize) {
                    let board = flop.add_card(board1 as usize).add_card(board2 as usize);
                    let mut strength = [
                        Vec::with_capacity(private_hand_cards[0].len()),
                        Vec::with_capacity(private_hand_cards[1].len()),
                    ];

                    for player in 0..2 {
                        strength[player] = private_hand_cards[player]
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &(hand1, hand2))| {
                                let (hand1, hand2) = (hand1 as usize, hand2 as usize);
                                if board.contains(hand1) || board.contains(hand2) {
                                    None
                                } else {
                                    let hand = board.add_card(hand1).add_card(hand2);
                                    Some((hand.evaluate() as usize, i))
                                }
                            })
                            .collect();

                        strength[player].shrink_to_fit();
                        strength[player].sort_unstable();
                    }

                    self.hand_strength[card_pair_index(board1, board2)] = strength;
                }
            }
        }
    }

    /// Initializes the root node of game tree.
    fn init_root(&mut self) {
        let current_memory_usage = AtomicU64::new(mem::size_of::<PostFlopNode>() as u64);
        let num_storage_elements = AtomicU64::new(0);
        let max_stack_size = [AtomicUsize::new(0), AtomicUsize::new(0)];

        let info = BuildTreeInfo {
            last_action: Action::None,
            last_bet: [0, 0],
            allin_flag: false,
            current_memory_usage: &current_memory_usage,
            num_storage_elements: &num_storage_elements,
            stack_size: [0, 0],
            max_stack_size: &max_stack_size,
        };

        self.root().children.clear();
        self.build_tree_recursive(&mut self.root(), &info);

        let stack_size = cmp::max(
            max_stack_size[0].load(Ordering::Relaxed),
            max_stack_size[1].load(Ordering::Relaxed),
        );

        #[cfg(feature = "custom_alloc")]
        STACK_UNIT_SIZE.store(4 * stack_size, Ordering::Relaxed);

        let current_memory_usage = current_memory_usage.load(Ordering::Relaxed);
        let num_storage_elements = num_storage_elements.load(Ordering::Relaxed);

        #[cfg(feature = "rayon")]
        let stack_usage = (4 * stack_size * rayon::current_num_threads()) as u64;
        #[cfg(not(feature = "rayon"))]
        let stack_usage = 4 * stack_size as u64;

        let mut memory_usage = current_memory_usage + stack_usage;

        memory_usage += vec_memory_usage(&self.hand_strength);
        for i in 0..2 {
            memory_usage += vec_memory_usage(&self.initial_weight[i]);
            memory_usage += vec_memory_usage(&self.private_hand_cards[i]);
            memory_usage += vec_memory_usage(&self.same_hand_index[i]);
            for strength in &self.hand_strength {
                memory_usage += vec_memory_usage(&strength[i]);
            }
        }

        self.is_memory_allocated = false;
        self.num_storage_elements = num_storage_elements;
        self.memory_usage = memory_usage + 2 * 4 * num_storage_elements;
        self.memory_usage_compressed = memory_usage + 2 * 2 * num_storage_elements;

        self.clear_storage();
    }

    /// Clears the storage.
    #[inline]
    fn clear_storage(&mut self) {
        self.cum_regret.lock().clear();
        self.cum_regret.lock().shrink_to_fit();
        self.strategy.lock().clear();
        self.strategy.lock().shrink_to_fit();
        self.cum_regret_compressed.lock().clear();
        self.cum_regret_compressed.lock().shrink_to_fit();
        self.strategy_compressed.lock().clear();
        self.strategy_compressed.lock().shrink_to_fit();
    }

    /// Builds the game tree recursively.
    fn build_tree_recursive(&self, node: &mut PostFlopNode, info: &BuildTreeInfo) {
        if node.is_terminal() {
            for i in 0..2 {
                atomic_set_max(&info.max_stack_size[i], info.stack_size[i]);
            }
            return;
        }

        // chance node
        if node.is_chance() {
            self.push_chances(node, info);

            let mut stack_size = info.stack_size;
            let f32_size = mem::size_of::<f32>();
            let col_size = f32_size * node.num_actions();
            for i in 0..2 {
                stack_size[i] += align_up(col_size * self.num_private_hands(i));
                stack_size[i] += align_up(f32_size * self.num_private_hands(i ^ 1));
            }

            for (action, child) in &node.children {
                self.build_tree_recursive(
                    &mut child.lock(),
                    &BuildTreeInfo {
                        last_action: *action,
                        last_bet: [0, 0],
                        stack_size,
                        ..*info
                    },
                );
            }
        }
        // player node
        else {
            self.push_actions(node, info);

            let mut stack_size = info.stack_size;
            let col_size = mem::size_of::<f32>() * node.num_actions();
            for i in 0..2 {
                let n = if i == node.player() { 2 } else { 1 };
                stack_size[i] += align_up(col_size * self.num_private_hands(i));
                stack_size[i] += n * align_up(col_size * self.num_private_hands(node.player()));
            }

            for (action, child) in &node.children {
                let mut last_bet = info.last_bet;
                let mut allin_flag = info.allin_flag;

                match *action {
                    Action::Call => {
                        last_bet[node.player as usize] = last_bet[node.player as usize ^ 1];
                    }
                    Action::Bet(size) | Action::Raise(size) => {
                        last_bet[node.player as usize] = size;
                    }
                    Action::AllIn(size) => {
                        last_bet[node.player as usize] = size;
                        allin_flag = true;
                    }
                    _ => {}
                }

                self.build_tree_recursive(
                    &mut child.lock(),
                    &BuildTreeInfo {
                        last_action: *action,
                        last_bet,
                        allin_flag,
                        stack_size,
                        ..*info
                    },
                )
            }
        }
    }

    /// Pushes the chance actions to the `node`.
    fn push_chances(&self, node: &mut PostFlopNode, info: &BuildTreeInfo) {
        let flop = self.config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        // deal turn
        if node.turn == NOT_DEALT {
            let next_player = if !info.allin_flag {
                PLAYER_OOP
            } else {
                PLAYER_CHANCE
            };

            node.children.reserve(49);

            for card in 0..52 {
                if (1 << card) & flop_mask == 0 && !self.turn_isomorphism_card.contains(&card) {
                    node.children.push((
                        Action::Chance(card),
                        MutexLike::new(PostFlopNode {
                            player: next_player,
                            turn: card,
                            amount: node.amount,
                            ..Default::default()
                        }),
                    ));
                }
            }
        }
        // deal river
        else {
            let turn_mask = flop_mask | (1 << node.turn);

            let next_player = if !info.allin_flag {
                PLAYER_OOP
            } else {
                PLAYER_TERMINAL_FLAG
            };

            node.children.reserve(48);

            for card in 0..52 {
                if (1 << card) & turn_mask == 0
                    && !self.river_isomorphism_card[node.turn as usize].contains(&card)
                {
                    node.children.push((
                        Action::Chance(card),
                        MutexLike::new(PostFlopNode {
                            player: next_player,
                            turn: node.turn,
                            river: card,
                            amount: node.amount,
                            ..Default::default()
                        }),
                    ));
                }
            }
        }

        node.children.shrink_to_fit();
        info.current_memory_usage
            .fetch_add(vec_memory_usage(&node.children), Ordering::Relaxed);
    }

    /// Pushes the actions to the `node`.
    fn push_actions(&self, node: &mut PostFlopNode, info: &BuildTreeInfo) {
        let player = node.player;
        let player_opponent = node.player ^ 1;

        let player_bet = info.last_bet[player as usize];
        let opponent_bet = info.last_bet[player_opponent as usize];

        let bet_diff = opponent_bet - player_bet;
        let pot = self.config.starting_pot + 2 * (node.amount + bet_diff);

        let max_bet = self.config.effective_stack - node.amount + player_bet;
        let min_bet = max_bet.min(opponent_bet + bet_diff);

        let (candidates, is_river) = if node.turn == NOT_DEALT {
            (&self.config.flop_bet_sizes, false)
        } else if node.river == NOT_DEALT {
            (&self.config.turn_bet_sizes, false)
        } else {
            (&self.config.river_bet_sizes, true)
        };

        let player_after_call = if is_river {
            PLAYER_TERMINAL_FLAG
        } else {
            PLAYER_CHANCE
        };

        let player_after_check = if player == PLAYER_OOP {
            player_opponent
        } else {
            player_after_call
        };

        let mut actions = Vec::new();

        if matches!(
            info.last_action,
            Action::None | Action::Check | Action::Chance(_)
        ) {
            // add check
            actions.push((Action::Check, player_after_check));

            // add first bet
            for &bet_size in &candidates[player as usize].bet {
                match bet_size {
                    BetSize::PotRelative(ratio) => {
                        let size = (pot as f32 * ratio).round() as i32;
                        actions.push((Action::Bet(size), player_opponent));
                    }
                    BetSize::LastBetRelative(_) => panic!("unexpected bet size"),
                }
            }

            // add all-in
            if max_bet <= (pot as f32 * self.config.add_all_in_threshold) as i32 {
                actions.push((Action::AllIn(max_bet), player_opponent));
            }
        } else {
            // add fold
            actions.push((Action::Fold, PLAYER_FOLD_FLAG | player));

            // add call
            actions.push((Action::Call, player_after_call));

            if !info.allin_flag {
                // add raise
                for &bet_size in &candidates[player as usize].raise {
                    match bet_size {
                        BetSize::PotRelative(ratio) => {
                            let size = opponent_bet + (pot as f32 * ratio).round() as i32;
                            actions.push((Action::Raise(size), player_opponent));
                        }
                        BetSize::LastBetRelative(ratio) => {
                            let size = (opponent_bet as f32 * ratio).round() as i32;
                            actions.push((Action::Raise(size), player_opponent));
                        }
                    }
                }

                // add all-in
                let all_in_threshold = (pot as f32 * self.config.add_all_in_threshold) as i32;
                if max_bet <= opponent_bet + all_in_threshold {
                    actions.push((Action::AllIn(max_bet), player_opponent));
                }
            }
        }

        let adjust_size = |size: i32| {
            let new_bet_diff = size - opponent_bet;
            let new_pot = pot + 2 * new_bet_diff;

            if max_bet <= size + (new_pot as f32 * self.config.replace_all_in_threshold) as i32 {
                return max_bet;
            }

            if !self.config.adjust_last_two_bet_sizes {
                return size;
            }

            let mut min_opponent_ratio = f32::INFINITY;
            for &bet_size in &candidates[player_opponent as usize].raise {
                match bet_size {
                    BetSize::PotRelative(ratio) => {
                        min_opponent_ratio = min_opponent_ratio.min(ratio);
                    }
                    BetSize::LastBetRelative(ratio) => {
                        let pot_ratio = size as f32 * (ratio - 1.0) / new_pot as f32;
                        min_opponent_ratio = min_opponent_ratio.min(pot_ratio);
                    }
                }
            }

            let min_opponent_bet = size + (new_pot as f32 * min_opponent_ratio).round() as i32;
            let next_bet_diff = min_opponent_bet - size;
            let next_pot = new_pot + 2 * next_bet_diff;

            // next opponent bet will be always all-in
            let threshold = (next_pot as f32 * self.config.replace_all_in_threshold) as i32;
            if max_bet <= min_opponent_bet + threshold {
                let ratio = new_bet_diff as f32 / pot as f32;
                let a = 2.0 * pot as f32 * ratio * min_opponent_ratio;
                let b = pot as f32 * (ratio + min_opponent_ratio);
                let c = (max_bet - opponent_bet) as f32;
                let coef = ((4.0 * a * c + b * b).sqrt() - b) / (2.0 * a);
                return opponent_bet + (new_bet_diff as f32 * coef).round() as i32;
            }

            size
        };

        // adjust bet sizes
        for (action, _) in actions.iter_mut() {
            match *action {
                Action::Bet(size) => {
                    let adjusted_size = adjust_size(size).clamp(min_bet, max_bet);
                    if adjusted_size == max_bet {
                        *action = Action::AllIn(max_bet);
                    } else if size != adjusted_size {
                        *action = Action::Bet(adjusted_size);
                    }
                }
                Action::Raise(size) => {
                    let adjusted_size = adjust_size(size).clamp(min_bet, max_bet);
                    if adjusted_size == max_bet {
                        *action = Action::AllIn(max_bet);
                    } else if size != adjusted_size {
                        *action = Action::Raise(adjusted_size);
                    }
                }
                _ => {}
            }
        }

        // remove duplicates
        actions.sort_unstable();
        actions.dedup();

        // push actions
        for (action, next_player) in actions {
            let mut amount = node.amount;
            if matches!(
                action,
                Action::Call | Action::Bet(_) | Action::Raise(_) | Action::AllIn(_)
            ) {
                amount += bet_diff;
            }

            node.children.push((
                action,
                MutexLike::new(PostFlopNode {
                    player: next_player,
                    turn: node.turn,
                    river: node.river,
                    amount,
                    ..Default::default()
                }),
            ));
        }

        node.children.shrink_to_fit();
        info.current_memory_usage
            .fetch_add(vec_memory_usage(&node.children), Ordering::Relaxed);

        let num_elems = node.num_actions() * self.num_private_hands(player as usize);
        node.num_elements = num_elems;
        info.num_storage_elements
            .fetch_add(num_elems as u64, Ordering::Relaxed);
    }

    /// Allocates memory recursively.
    fn allocate_memory_recursive(&self, node: &mut PostFlopNode, counter: &AtomicUsize) {
        if node.is_terminal() {
            return;
        }

        if !node.is_chance() {
            let index = counter.fetch_add(node.num_elements, Ordering::SeqCst);
            unsafe {
                if self.is_compression_enabled() {
                    let cum_regret_ptr = self.cum_regret_compressed.lock().as_mut_ptr();
                    let strategy_ptr = self.strategy_compressed.lock().as_mut_ptr();
                    node.cum_regret = ptr::null_mut();
                    node.strategy = ptr::null_mut();
                    node.cum_regret_compressed = cum_regret_ptr.add(index);
                    node.strategy_compressed = strategy_ptr.add(index);
                } else {
                    let cum_regret_ptr = self.cum_regret.lock().as_mut_ptr();
                    let strategy_ptr = self.strategy.lock().as_mut_ptr();
                    node.cum_regret = cum_regret_ptr.add(index);
                    node.strategy = strategy_ptr.add(index);
                    node.cum_regret_compressed = ptr::null_mut();
                    node.strategy_compressed = ptr::null_mut();
                }
            }
        }

        for action in node.actions() {
            self.allocate_memory_recursive(&mut node.play(action), counter);
        }
    }
}

impl GameNode for PostFlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player == PLAYER_CHANCE
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.children.len()
    }

    #[inline]
    fn chance_factor(&self) -> f32 {
        [1.0 / 45.0, 1.0 / 44.0][(self.turn != NOT_DEALT) as usize]
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children[action].1.lock()
    }

    #[inline]
    fn cum_regret(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.cum_regret, self.num_elements) }
    }

    #[inline]
    fn cum_regret_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.cum_regret, self.num_elements) }
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.strategy, self.num_elements) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.strategy, self.num_elements) }
    }

    #[inline]
    fn cum_regret_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.cum_regret_compressed, self.num_elements) }
    }

    #[inline]
    fn cum_regret_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.cum_regret_compressed, self.num_elements) }
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.strategy_compressed, self.num_elements) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.strategy_compressed, self.num_elements) }
    }

    #[inline]
    fn cum_regret_scale(&self) -> f32 {
        self.cum_regret_scale
    }

    #[inline]
    fn set_cum_regret_scale(&mut self, scale: f32) {
        self.cum_regret_scale = scale;
    }

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.strategy_scale
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.strategy_scale = scale;
    }

    #[inline]
    fn is_strategy_locked(&self) -> bool {
        self.is_strategy_locked
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }
}

impl PostFlopNode {
    /// Returns the betted amount.
    #[inline]
    pub fn amount(&self) -> i32 {
        self.amount
    }

    /// Returns the possible actions for the current player.
    #[inline]
    pub fn get_actions(&self) -> Vec<Action> {
        self.children.iter().map(|(action, _)| *action).collect()
    }
}

impl Default for PostFlopNode {
    #[inline]
    fn default() -> Self {
        Self {
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            amount: 0,
            children: Vec::new(),
            cum_regret: ptr::null_mut(),
            strategy: ptr::null_mut(),
            cum_regret_compressed: ptr::null_mut(),
            strategy_compressed: ptr::null_mut(),
            cum_regret_scale: 0.0,
            strategy_scale: 0.0,
            num_elements: 0,
            is_strategy_locked: false,
        }
    }
}

impl Default for GameConfig {
    #[inline]
    fn default() -> Self {
        Self {
            flop: [NOT_DEALT; 3],
            starting_pot: 0,
            effective_stack: 0,
            range: Default::default(),
            flop_bet_sizes: Default::default(),
            turn_bet_sizes: Default::default(),
            river_bet_sizes: Default::default(),
            add_all_in_threshold: 1.5,
            replace_all_in_threshold: 0.1,
            adjust_last_two_bet_sizes: true,
        }
    }
}

#[inline]
fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * mem::size_of::<T>() as u64
}

/// Attempts to convert an optionally space-separated string into a sorted flop array.
///
/// # Examples
/// ```
/// use postflop_solver::flop_from_str;
///
/// let flop = flop_from_str("2c 3d 4h");
///
/// assert_eq!(flop, Ok([0, 5, 10]));
/// ```
pub fn flop_from_str(s: &str) -> Result<[u8; 3], String> {
    let mut result = [0; 3];
    let mut chars = s.chars();

    result[0] = card_from_str(&mut chars)?;
    result[1] = card_from_str(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;
    result[2] = card_from_str(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;

    if chars.next().is_some() {
        return Err("expected three cards".to_string());
    }

    result.sort_unstable();

    if result[0] == result[1] || result[1] == result[2] {
        return Err("cards must be unique".to_string());
    }

    Ok(result)
}

#[inline]
fn card_from_str<T: Iterator<Item = char>>(chars: &mut T) -> Result<u8, String> {
    let rank_char = chars.next().ok_or_else(|| "parse failed".to_string())?;
    let suit_char = chars.next().ok_or_else(|| "parse failed".to_string())?;

    let rank = match rank_char {
        'A' => 12,
        'K' => 11,
        'Q' => 10,
        'J' => 9,
        'T' => 8,
        '2'..='9' => rank_char as u8 - b'2',
        _ => return Err(format!("expected rank: {rank_char}")),
    };

    let suit = match suit_char {
        's' => 3,
        'h' => 2,
        'd' => 1,
        'c' => 0,
        _ => return Err(format!("expected suit: {suit_char}")),
    };

    Ok((rank << 2) | suit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::*;
    use crate::utility::*;

    #[test]
    fn test_flop_from_str() {
        let tests = [("Qs Jh 2h", [2, 38, 43]), ("Td9d6h", [18, 29, 33])];
        for test in tests {
            assert_eq!(flop_from_str(test.0).unwrap(), test.1);
        }
    }

    #[test]
    fn all_check_all_range() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: [Range::ones(); 2],
            ..Default::default()
        };
        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        normalize_strategy(&game);
        compute_ev(&game);
        let ev0 = compute_ev_scalar(&game, &game.root()) + 30.0;
        let ev1 = 60.0 - ev0;
        assert!((ev0 - 30.0).abs() < 1e-4);
        assert!((ev1 - 30.0).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: [Range::ones(); 2],
            river_bet_sizes: [bet_sizes_from_str("50%", "").unwrap(), Default::default()],
            ..Default::default()
        };
        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        normalize_strategy(&game);
        compute_ev(&game);
        let ev0 = compute_ev_scalar(&game, &game.root()) + 30.0;
        let ev1 = 60.0 - ev0;
        assert!((ev0 - 37.5).abs() < 1e-4);
        assert!((ev1 - 22.5).abs() < 1e-4);
    }

    #[test]
    fn always_win() {
        // be careful for straight flushes
        let lose_range_str = "22+,A2+,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let config = GameConfig {
            flop: flop_from_str("AcAdKh").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
            ..Default::default()
        };
        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        normalize_strategy(&game);
        compute_ev(&game);
        let ev0 = compute_ev_scalar(&game, &game.root()) + 30.0;
        let ev1 = 60.0 - ev0;
        assert!((ev0 - 60.0).abs() < 1e-4);
        assert!((ev1 - 0.0).abs() < 1e-4);
    }

    #[test]
    fn no_assignment() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: ["TT".parse().unwrap(), "TT".parse().unwrap()],
            ..Default::default()
        };
        let game = PostFlopGame::with_config(&config);
        assert!(game.is_err());
    }

    #[test]
    #[ignore]
    fn cfr_solve1() {
        // top-40% range
        let oop_range =
            "22+,A2s+,A8o+,K7s+,K9o+,Q8s+,Q9o+,J8s+,J9o+,T8+,97+,86+,75+,64s+,65o,54,43s";

        // top-25% range
        let ip_range = "22+,A4s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9+,T9,98s,87s,76s,65s";

        let bet_sizes = bet_sizes_from_str("50%", "50%").unwrap();

        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            starting_pot: 60,
            effective_stack: 770,
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            add_all_in_threshold: 0.0,
            replace_all_in_threshold: 0.0,
            adjust_last_two_bet_sizes: false,
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        println!(
            "memory usage: {:.2}GB",
            game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        game.allocate_memory(false);

        solve(&game, 1000, 60.0 * 0.005, true);
        compute_ev(&game);
        let ev0 = compute_ev_scalar(&game, &game.root()) + 30.0;
        let ev1 = 60.0 - ev0;

        // verified by GTO+
        assert!((ev0 - 26.24).abs() < 0.5);
        assert!((ev1 - 33.76).abs() < 0.5);
    }

    #[test]
    #[ignore]
    fn cfr_solve2() {
        let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
        let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

        let flop_bet_sizes = bet_sizes_from_str("52%", "45%").unwrap();
        let turn_bet_sizes = bet_sizes_from_str("55%", "45%").unwrap();
        let river_bet_sizes = bet_sizes_from_str("70%", "45%").unwrap();

        let config = GameConfig {
            flop: flop_from_str("QsJh2h").unwrap(),
            starting_pot: 180,
            effective_stack: 910,
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop_bet_sizes: [flop_bet_sizes.clone(), flop_bet_sizes.clone()],
            turn_bet_sizes: [turn_bet_sizes.clone(), turn_bet_sizes.clone()],
            river_bet_sizes: [river_bet_sizes.clone(), river_bet_sizes.clone()],
            add_all_in_threshold: 5.0,
            replace_all_in_threshold: 0.1,
            adjust_last_two_bet_sizes: false,
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        println!(
            "memory usage: {:.2}GB",
            game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        game.allocate_memory(false);

        solve(&game, 1000, 180.0 * 0.0035, true);
        compute_ev(&game);
        let ev0 = compute_ev_scalar(&game, &game.root()) + 90.0;
        let ev1 = 180.0 - ev0;

        // verified by PioSolver
        assert!((ev0 - 105.0).abs() < 0.5);
        assert!((ev1 - 75.0).abs() < 0.5);
    }
}
