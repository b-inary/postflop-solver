use crate::bet_size::*;
use crate::interface::*;
use crate::mutex_like::*;
use crate::range::*;
use crate::sliceop::*;
use crate::utility::*;
use std::cmp;
use std::mem;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;
#[cfg(feature = "custom-alloc")]
use std::vec;

#[cfg(not(feature = "holdem-hand-evaluator"))]
use crate::hand::Hand;
#[cfg(feature = "holdem-hand-evaluator")]
use holdem_hand_evaluator::Hand;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StrengthItem {
    strength: usize,
    index: usize,
}

type SwapList = [Vec<(usize, usize)>; 2];

/// A struct representing a postflop game.
pub struct PostFlopGame {
    // Postflop game configuration.
    config: GameConfig,

    // computed from `config`
    root: Box<MutexLike<PostFlopNode>>,
    num_combinations_inv: f64,
    initial_weight: [Vec<f32>; 2],
    private_hand_cards: [Vec<(u8, u8)>; 2],
    same_hand_index: [Vec<Option<usize>>; 2],
    hand_strength: Vec<[Vec<StrengthItem>; 2]>,
    turn_isomorphism: Vec<usize>,
    turn_isomorphism_cards: Vec<u8>,
    turn_isomorphism_swap: [SwapList; 4],
    river_isomorphism: Vec<Vec<usize>>,
    river_isomorphism_cards: Vec<Vec<u8>>,
    river_isomorphism_swap: Vec<[SwapList; 4]>,

    // store options
    is_memory_allocated: bool,
    is_compression_enabled: bool,
    misc_memory_usage: u64,
    num_storage_elements: u64,

    // global storage
    storage1: MutexLike<Vec<f32>>,
    storage2: MutexLike<Vec<f32>>,
    storage1_compressed: MutexLike<Vec<u16>>,
    storage2_compressed: MutexLike<Vec<i16>>,

    // result interpreter
    is_solved: bool,
    node: *const PostFlopNode,
    turn: u8,
    river: u8,
    weights: [Vec<f32>; 2],
    normalized_weights: [Vec<f32>; 2],
    normalized_weights_cached: bool,
    normalize_factor: f32,
    turn_swapped_suit: Option<(u8, u8)>,
    turn_swap: *const SwapList,
    river_swap: *const SwapList,
}

unsafe impl Send for PostFlopGame {}
unsafe impl Sync for PostFlopGame {}

/// A struct representing a node in postflop game tree.
pub struct PostFlopNode {
    player: u16,
    turn: u8,
    river: u8,
    amount: i32,
    children: Vec<(Action, MutexLike<PostFlopNode>)>,
    storage1: *mut u8,
    storage2: *mut u8,
    scale1: f32,
    scale2: f32,
    num_elements: usize,
}

unsafe impl Send for PostFlopNode {}
unsafe impl Sync for PostFlopNode {}

/// A struct for postflop game configuration.
///
/// # Examples
/// ```
/// use postflop_solver::*;
///
/// let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
/// let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
/// let bet_sizes = BetSizeCandidates::try_from(("50%", "50%")).unwrap();
///
/// let config = GameConfig {
///     flop: flop_from_str("Td9d6h").unwrap(),
///     turn: NOT_DEALT, // or `card_from_str("As").unwrap()`
///     river: NOT_DEALT,
///     starting_pot: 200,
///     effective_stack: 900,
///     range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
///     flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     add_all_in_threshold: 1.2,
///     force_all_in_threshold: 0.1,
///     adjust_last_two_bet_sizes: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct GameConfig {
    /// Flop cards: each card must be unique and in range [`0`, `52`).
    pub flop: [u8; 3],

    /// Turn card: must be in range [`0`, `52`) or `NOT_DEALT`.
    pub turn: u8,

    /// River card: must be in range [`0`, `52`) or `NOT_DEALT`.
    pub river: u8,

    /// Initial pot size: must be greater than `0`.
    pub starting_pot: i32,

    /// Initial effective stack size: must be greater than `0`.
    pub effective_stack: i32,

    /// Initial range of each player.
    pub range: [Range; 2],

    /// Bet size candidates of each player in flop.
    pub flop_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player in turn.
    pub turn_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player in river.
    pub river_bet_sizes: [BetSizeCandidates; 2],

    /// Add all-in action when SPR is below this value (set `0.0` to disable).
    pub add_all_in_threshold: f32,

    /// Force all-in action when the ratio of opponent's next raise size to the pot size will be
    /// less than this value (set `0.0` to disable).
    pub force_all_in_threshold: f32,

    /// Enable bet size adjustment of last two bets.
    pub adjust_last_two_bet_sizes: bool,
}

/// Available actions in a postflop game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    /// Only used for the previous action of the root node.
    None,

    /// Fold action.
    Fold,

    /// Check action.
    Check,

    /// Call action.
    Call,

    /// Bet action with a specified amount.
    Bet(i32),

    /// Raise action with a specified amount.
    Raise(i32),

    /// All-in action with a specified amount.
    AllIn(i32),

    /// Chance action with a card ID (in range [`0`, `52`)).
    Chance(u8),
}

struct BuildTreeInfo<'a> {
    last_action: Action,
    last_bet: [i32; 2],
    allin_flag: bool,
    current_memory_usage: &'a AtomicU64,
    num_storage_elements: &'a AtomicU64,
    stack_size: [usize; 2],
    max_stack_size: &'a [AtomicUsize; 2],
}

const PLAYER_OOP: u16 = 0;
const PLAYER_CHANCE: u16 = 0xff;
const PLAYER_MASK: u16 = 0xff;
const PLAYER_TERMINAL_FLAG: u16 = 0x100;
const PLAYER_FOLD_FLAG: u16 = 0x300;

/// Constant representing that the card is not yet dealt.
pub const NOT_DEALT: u8 = 0xff;

#[cfg(not(feature = "custom-alloc"))]
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

/// Computes the average with given weights.
#[inline]
pub fn compute_average(slice: &[f32], weights: &[f32]) -> f32 {
    let mut weight_sum = 0.0;
    let mut product_sum = 0.0;
    for (&v, &w) in slice.iter().zip(weights.iter()) {
        weight_sum += w as f64;
        product_sum += v as f64 * w as f64;
    }
    (product_sum / weight_sum) as f32
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
                    let StrengthItem { strength, index } = *player_strength.get_unchecked(i);
                    while j < opponent_len && opponent_strength.get_unchecked(j).strength < strength
                    {
                        let opponent_index = opponent_strength.get_unchecked(j).index;
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
                    let StrengthItem { strength, index } = *player_strength.get_unchecked(i);
                    while j > 0 && opponent_strength.get_unchecked(j - 1).strength > strength {
                        let opponent_index = opponent_strength.get_unchecked(j - 1).index;
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
            &self.turn_isomorphism_swap[self.turn_isomorphism_cards[index] as usize & 3]
        } else {
            &self.river_isomorphism_swap[node.turn as usize]
                [self.river_isomorphism_cards[node.turn as usize][index] as usize & 3]
        }
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.is_memory_allocated
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.is_solved
    }

    #[inline]
    fn set_solved(&mut self) {
        self.is_solved = true;
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }
}

impl PostFlopGame {
    /// Constructs a new empty [`PostFlopGame`] (needs `update_config()` before solving).
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

    /// Updates the game configuration. The solved result will be lost.
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
        let uncompressed = self.misc_memory_usage + 8 * self.num_storage_elements;
        let compressed = self.misc_memory_usage + 4 * self.num_storage_elements;
        (uncompressed, compressed)
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
            self.storage1_compressed = MutexLike::new(vec![0; num_elems]);
            self.storage2_compressed = MutexLike::new(vec![0; num_elems]);
        } else {
            self.storage1 = MutexLike::new(vec![0.0; num_elems]);
            self.storage2 = MutexLike::new(vec![0.0; num_elems]);
        }

        let counter = AtomicUsize::new(0);
        self.allocate_memory_recursive(&mut self.root(), &counter);
    }

    /// Returns a card list of isomorphic chances.
    #[inline]
    fn isomorphic_cards(&self, node: &PostFlopNode) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.turn_isomorphism_cards
        } else {
            &self.river_isomorphism_cards[node.turn as usize]
        }
    }

    /// Checks the configuration for errors.
    fn check_config(&mut self) -> Result<(), String> {
        let flop = &self.config.flop;

        if flop.contains(&NOT_DEALT) {
            return Err("Flop cards not initialized".to_string());
        }

        if flop.iter().any(|&c| 52 <= c) {
            return Err(format!("Flop cards must be in [0, 52): flop = {:?}", flop));
        }

        if flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2] {
            return Err(format!("Flop cards must be unique: flop = {:?}", flop));
        }

        if self.config.turn != NOT_DEALT {
            if 52 <= self.config.turn {
                return Err(format!(
                    "Turn card must be in [0, 52): turn = {}",
                    self.config.turn
                ));
            }

            if flop.contains(&self.config.turn) {
                return Err(format!(
                    "Turn card must be different from flop cards: turn = {}",
                    self.config.turn
                ));
            }
        }

        if self.config.river != NOT_DEALT {
            if 52 <= self.config.river {
                return Err(format!(
                    "River card must be in [0, 52): river = {}",
                    self.config.river
                ));
            }

            if flop.contains(&self.config.river) {
                return Err(format!(
                    "River card must be different from flop cards: river = {}",
                    self.config.river
                ));
            }

            if self.config.turn == self.config.river {
                return Err(format!(
                    "River card must be different from turn card: river = {}",
                    self.config.river
                ));
            }

            if self.config.turn == NOT_DEALT {
                return Err(format!(
                    "River card specified without turn card: river = {}",
                    self.config.river
                ));
            }
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

        if self.config.range[0].is_empty() {
            return Err("OOP range is empty".to_string());
        }

        if self.config.range[1].is_empty() {
            return Err("IP range is empty".to_string());
        }

        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if self.config.turn != NOT_DEALT {
            board_mask |= 1 << self.config.turn;
        }
        if self.config.river != NOT_DEALT {
            board_mask |= 1 << self.config.river;
        }

        let mut num_combinations = 0.0;
        for c1 in 0..52 {
            for c2 in c1 + 1..52 {
                let oop_mask: u64 = (1 << c1) | (1 << c2);
                let oop_weight = self.config.range[0].get_weight_by_cards(c1, c2);
                if oop_mask & board_mask == 0 && oop_weight > 0.0 {
                    for c3 in 0..52 {
                        for c4 in c3 + 1..52 {
                            let ip_mask: u64 = (1 << c3) | (1 << c4);
                            let ip_weight = self.config.range[1].get_weight_by_cards(c3, c4);
                            if ip_mask & (board_mask | oop_mask) == 0 {
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

        // interpreter
        self.is_solved = false;
        self.back_to_root();
        self.normalized_weights = [
            vec![0.0; self.num_private_hands(0)],
            vec![0.0; self.num_private_hands(1)],
        ];
    }

    /// Initializes fields `initial_weight`, `private_hand_cards` and `same_hand_index`.
    fn init_range(&mut self) {
        let flop = &self.config.flop;
        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if self.config.turn != NOT_DEALT {
            board_mask |= 1 << self.config.turn;
        }
        if self.config.river != NOT_DEALT {
            board_mask |= 1 << self.config.river;
        }

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
                    if weight > 0.0 && hand_mask & board_mask == 0 {
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
                if range[0].is_suit_isomorphic(suit1, suit2)
                    && range[1].is_suit_isomorphic(suit1, suit2)
                {
                    suit_isomorphism[suit2 as usize] = suit_isomorphism[suit1 as usize];
                    continue 'outer;
                }
            }
            suit_isomorphism[suit2 as usize] = next_index;
            next_index += 1;
        }

        let flop = &self.config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let mut flop_rankset = [0; 4];
        for &card in flop {
            let rank = card >> 2;
            let suit = card & 3;
            flop_rankset[suit as usize] |= 1 << rank;
        }

        self.turn_isomorphism.clear();
        self.turn_isomorphism_cards.clear();
        self.turn_isomorphism_swap.iter_mut().for_each(|x| {
            x[0].clear();
            x[1].clear();
        });

        let mut isomorphic_suit = [None; 4];
        let mut reverse_table = [usize::MAX; 52 * 51 / 2];

        // turn isomorphism
        if self.config.turn == NOT_DEALT {
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
                    self.turn_isomorphism_cards.push(card);
                } else {
                    indices[card as usize] = counter;
                    counter += 1;
                }
            }
        }

        self.river_isomorphism.clear();
        self.river_isomorphism_cards.clear();
        self.river_isomorphism_swap.clear();

        // river isomorphism
        if self.config.river == NOT_DEALT {
            for turn in 0..52 {
                self.river_isomorphism.push(Vec::new());
                self.river_isomorphism_cards.push(Vec::new());
                self.river_isomorphism_swap.push(Default::default());

                if (1 << turn) & flop_mask != 0
                    || (self.config.turn != NOT_DEALT && turn != self.config.turn)
                {
                    continue;
                }

                let river_isomorphism = self.river_isomorphism.last_mut().unwrap();
                let river_isomorphism_cards = self.river_isomorphism_cards.last_mut().unwrap();
                let river_isomorphism_swap = self.river_isomorphism_swap.last_mut().unwrap();

                let turn_mask = flop_mask | (1 << turn);
                let mut turn_rankset = flop_rankset;
                turn_rankset[turn as usize & 3] |= 1 << (turn >> 2);

                isomorphic_suit.fill(None);

                for suit1 in 1..4 {
                    for suit2 in 0..suit1 {
                        if (flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                            || self.config.turn != NOT_DEALT)
                            && turn_rankset[suit1 as usize] == turn_rankset[suit2 as usize]
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

                let mut counter = 0;
                let mut indices = [0; 52];

                for card in 0..52 {
                    if (1 << card) & turn_mask != 0 {
                        continue;
                    }

                    let suit = card & 3;

                    if let Some(replace_suit) = isomorphic_suit[suit as usize] {
                        let replace_card = card - suit + replace_suit;
                        river_isomorphism.push(indices[replace_card as usize]);
                        river_isomorphism_cards.push(card);
                    } else {
                        indices[card as usize] = counter;
                        counter += 1;
                    }
                }
            }
        }
    }

    /// Initializes a field `hand_strength`.
    fn init_hand_strength(&mut self) {
        let mut flop = Hand::new();
        for &card in &self.config.flop {
            flop = flop.add_card(card as usize);
        }

        self.hand_strength = vec![Default::default(); 52 * 51 / 2];
        let private_hand_cards = &self.private_hand_cards;

        for board1 in 0..52 {
            for board2 in board1 + 1..52 {
                let mut is_possible =
                    !flop.contains(board1 as usize) && !flop.contains(board2 as usize);
                if self.config.river != NOT_DEALT {
                    is_possible &= (self.config.turn == board1 && self.config.river == board2)
                        || (self.config.turn == board2 && self.config.river == board1);
                } else if self.config.turn != NOT_DEALT {
                    is_possible &= self.config.turn == board1 || self.config.turn == board2;
                }

                if is_possible {
                    let board = flop.add_card(board1 as usize).add_card(board2 as usize);
                    let mut strength = [
                        Vec::with_capacity(private_hand_cards[0].len()),
                        Vec::with_capacity(private_hand_cards[1].len()),
                    ];

                    for player in 0..2 {
                        strength[player] = private_hand_cards[player]
                            .iter()
                            .enumerate()
                            .filter_map(|(index, &(hand1, hand2))| {
                                let (hand1, hand2) = (hand1 as usize, hand2 as usize);
                                if board.contains(hand1) || board.contains(hand2) {
                                    None
                                } else {
                                    let hand = board.add_card(hand1).add_card(hand2);
                                    Some(StrengthItem {
                                        strength: hand.evaluate() as usize,
                                        index,
                                    })
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

        let mut root = self.root();
        *root = PostFlopNode::default();
        root.turn = self.config.turn;
        root.river = self.config.river;

        self.build_tree_recursive(&mut root, &info);

        let current_memory_usage = current_memory_usage.load(Ordering::Relaxed);
        let num_storage_elements = num_storage_elements.load(Ordering::Relaxed);

        let stack_size = cmp::max(
            max_stack_size[0].load(Ordering::Relaxed),
            max_stack_size[1].load(Ordering::Relaxed),
        );

        #[cfg(feature = "custom-alloc")]
        STACK_UNIT_SIZE.store(4 * stack_size, Ordering::Relaxed);

        #[cfg(feature = "rayon")]
        let stack_usage = (4 * stack_size * rayon::current_num_threads()) as u64;
        #[cfg(not(feature = "rayon"))]
        let stack_usage = 4 * stack_size as u64;

        let mut memory_usage = current_memory_usage + stack_usage;

        memory_usage += vec_memory_usage(&self.hand_strength);
        memory_usage += vec_memory_usage(&self.turn_isomorphism);
        memory_usage += vec_memory_usage(&self.turn_isomorphism_cards);
        memory_usage += vec_memory_usage(&self.river_isomorphism);
        memory_usage += vec_memory_usage(&self.river_isomorphism_cards);
        memory_usage += vec_memory_usage(&self.river_isomorphism_swap);

        for i in 0..2 {
            memory_usage += vec_memory_usage(&self.initial_weight[i]);
            memory_usage += vec_memory_usage(&self.private_hand_cards[i]);
            memory_usage += vec_memory_usage(&self.same_hand_index[i]);
            for strength in &self.hand_strength {
                memory_usage += vec_memory_usage(&strength[i]);
            }
            for swap in &self.turn_isomorphism_swap {
                memory_usage += vec_memory_usage(&swap[i]);
            }
            for swap_list in &self.river_isomorphism_swap {
                for swap in swap_list {
                    memory_usage += vec_memory_usage(&swap[i]);
                }
            }
        }

        self.is_memory_allocated = false;
        self.misc_memory_usage = memory_usage;
        self.num_storage_elements = num_storage_elements;

        self.clear_storage();
    }

    /// Clears the storage.
    #[inline]
    fn clear_storage(&mut self) {
        self.storage1.lock().clear();
        self.storage2.lock().clear();
        self.storage1_compressed.lock().clear();
        self.storage2_compressed.lock().clear();
        self.storage1.lock().shrink_to_fit();
        self.storage2.lock().shrink_to_fit();
        self.storage1_compressed.lock().shrink_to_fit();
        self.storage2_compressed.lock().shrink_to_fit();
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
                if (1 << card) & flop_mask == 0 && !self.turn_isomorphism_cards.contains(&card) {
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
                    && !self.river_isomorphism_cards[node.turn as usize].contains(&card)
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
        let min_bet = (opponent_bet + bet_diff).clamp(1, max_bet);

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

            if max_bet <= size + (new_pot as f32 * self.config.force_all_in_threshold) as i32 {
                return max_bet;
            }

            if !self.config.adjust_last_two_bet_sizes {
                return size;
            }

            let mut min_opponent_ratio = f32::MIN;
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
            let threshold = (next_pot as f32 * self.config.force_all_in_threshold) as i32;
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
                if self.is_compression_enabled {
                    let storage1_ptr = self.storage1_compressed.lock().as_mut_ptr();
                    let storage2_ptr = self.storage2_compressed.lock().as_mut_ptr();
                    node.storage1 = storage1_ptr.add(index) as *mut u8;
                    node.storage2 = storage2_ptr.add(index) as *mut u8;
                } else {
                    let storage1_ptr = self.storage1.lock().as_mut_ptr();
                    let storage2_ptr = self.storage2.lock().as_mut_ptr();
                    node.storage1 = storage1_ptr.add(index) as *mut u8;
                    node.storage2 = storage2_ptr.add(index) as *mut u8;
                }
            }
        }

        for action in node.actions() {
            self.allocate_memory_recursive(&mut node.play(action), counter);
        }
    }

    /// Moves the current node back to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        self.node = &*self.root();
        self.turn = self.config.turn;
        self.river = self.config.river;
        self.weights = self.initial_weight.clone();
        self.normalized_weights_cached = false;
        self.normalize_factor = 1.0 / self.num_combinations_inv as f32;
        self.turn_swapped_suit = None;
        self.turn_swap = ptr::null();
        self.river_swap = ptr::null();
    }

    /// Returns the available actions for the current node.
    ///
    /// Note: If the current node is a chance node, isomorphic chances are grouped together into
    /// one representative action.
    #[inline]
    pub fn available_actions(&self) -> Vec<Action> {
        self.node().available_actions()
    }

    /// Returns whether the available actions are terminal.
    ///
    /// Note that the call action after the all-in action is considered as terminal.
    #[inline]
    pub fn is_terminal_action(&self) -> Vec<bool> {
        self.node()
            .actions()
            .map(|action| {
                let child = self.node().play(action);
                child.is_terminal() || child.amount == self.config.effective_stack
            })
            .collect()
    }

    /// Returns whether the current node is a chance node.
    #[inline]
    pub fn is_chance_node(&self) -> bool {
        self.node().is_chance()
    }

    /// If the current node is a chance node, returns a list of cards that may be dealt.
    ///
    /// The returned value is a 64-bit integer.
    /// The `i`-th bit is set to 1 if the card of ID `i` may be dealt.
    /// If the current node is not a chance node, returns `0`.
    ///
    /// Card ID: `"2c"` => `0`, `"2d"` => `1`, `"2h"` => `2`, ..., `"As"` => `51`.
    pub fn possible_cards(&self) -> u64 {
        if !self.node().is_chance() {
            return 0;
        }

        let flop = self.config.flop;
        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }

        let mut mask: u64 = (1 << 52) - 1;

        for (i, &(c1, c2)) in self.private_hand_cards[0].iter().enumerate() {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            let oop_weight = self.weights[0][i];
            if board_mask & oop_mask == 0 && oop_weight > 0.0 {
                for (j, &(c3, c4)) in self.private_hand_cards[1].iter().enumerate() {
                    let ip_mask: u64 = (1 << c3) | (1 << c4);
                    let ip_weight = self.weights[1][j];
                    if (board_mask | oop_mask) & ip_mask == 0 && ip_weight > 0.0 {
                        mask &= board_mask | oop_mask | ip_mask;
                    }
                }
                if mask == board_mask {
                    break;
                }
            }
        }

        ((1 << 52) - 1) ^ mask
    }

    /// Returns the current player (0 = OOP, 1 = IP).
    ///
    /// If the current node is a chance node, returns an undefined value.
    #[inline]
    pub fn current_player(&self) -> usize {
        self.node().player()
    }

    /// Plays the given action. Playing a terminal action is not allowed.
    /// - `action`
    ///   - If the current node is a chance node, `action` corresponds to the card ID of the dealt
    ///     card.
    ///   - If the current node is not a chance node, plays the `action`-th action of
    ///     `available_actions()`.
    pub fn play(&mut self, action: usize) {
        if !self.is_solved {
            panic!("game is not solved");
        }

        // chande node
        if self.is_chance_node() {
            let is_turn = self.turn == NOT_DEALT;
            let actual_card = action as u8;

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

            // finds the action index from available actions
            for (i, &action) in actions.iter().enumerate() {
                if action == Action::Chance(action_card) {
                    action_index = i;
                    break;
                }
            }

            // finds the action index from isomorphic chances
            if action_index == usize::MAX {
                let isomorphism = self.isomorphic_chances(self.node());
                let isomorphic_cards = self.isomorphic_cards(self.node());
                for (i, &repr_index) in isomorphism.iter().enumerate() {
                    if action_card == isomorphic_cards[i] {
                        action_index = repr_index;
                        if is_turn {
                            self.turn_swap = self.isomorphic_swap(self.node(), i);
                            if let Action::Chance(repr_card) = actions[repr_index] {
                                self.turn_swapped_suit = Some((action_card & 3, repr_card & 3));
                            }
                        } else {
                            self.river_swap = self.isomorphic_swap(self.node(), i);
                        }
                        break;
                    }
                }
            }

            // panics if the action is not found
            if action_index == usize::MAX {
                panic!("invalid action");
            }

            // updates the state
            self.node = &*self.node().play(action_index);
            if is_turn {
                self.turn = actual_card;
                self.normalize_factor *= 45.0;
            } else {
                self.river = actual_card;
                self.normalize_factor *= 44.0;
            }
        }
        // not chance node
        else {
            // panics if the action is invalid
            if action >= self.node().num_actions() {
                panic!("invalid action");
            }

            // updates the weights
            if self.node().num_actions() > 1 {
                let player = self.node().player();
                let strategy = self.strategy();
                let weights = row(&strategy, action, self.num_private_hands(player));
                mul_slice(&mut self.weights[player], weights);
            }

            // updates the node
            self.node = &*self.node().play(action);
        }

        if self.node().is_terminal() || self.node().amount == self.config.effective_stack {
            panic!("playing a terminal action is not allowed");
        }

        self.normalized_weights_cached = false;
    }

    /// Computes the normalized weights and caches them.
    ///
    /// After mutating the current node, this method must be called once before calling
    /// `normalized_weights()`, `equity()`, `expected_values()`, or `expected_values_detail()`.
    pub fn cache_normalized_weights(&mut self) {
        if self.normalized_weights_cached {
            return;
        }

        let mut normalized_weights_f64 = [
            vec![0.0; self.num_private_hands(0)],
            vec![0.0; self.num_private_hands(1)],
        ];

        let oop_hands = &self.private_hand_cards[0];
        let ip_hands = &self.private_hand_cards[1];

        let mut board_mask: u64 = 0;
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }
        if self.river != NOT_DEALT {
            board_mask |= 1 << self.river;
        }

        for (i, &(c1, c2)) in oop_hands.iter().enumerate() {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            let oop_weight = self.weights[0][i];
            if board_mask & oop_mask == 0 && oop_weight > 0.0 {
                for (j, &(c3, c4)) in ip_hands.iter().enumerate() {
                    let ip_mask: u64 = (1 << c3) | (1 << c4);
                    let ip_weight = self.weights[1][j];
                    if (board_mask | oop_mask) & ip_mask == 0 && ip_weight > 0.0 {
                        let weight = oop_weight as f64 * ip_weight as f64;
                        normalized_weights_f64[0][i] += weight;
                        normalized_weights_f64[1][j] += weight;
                    }
                }
            }
        }

        for player in 0..2 {
            self.normalized_weights[player]
                .iter_mut()
                .zip(normalized_weights_f64[player].iter())
                .for_each(|(w, &w_f64)| *w = w_f64 as f32);
        }

        self.normalized_weights_cached = true;
    }

    /// Returns the weights of each private hand of the given player.
    #[inline]
    pub fn weights(&self, player: usize) -> &[f32] {
        &self.weights[player]
    }

    /// Returns the normalized weights of each private hand of the given player.
    ///
    /// The "normalized weights" represent the actual number of combinations that the player is
    /// holding each hand.
    ///
    /// After mutating the current node, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    #[inline]
    pub fn normalized_weights(&self, player: usize) -> &[f32] {
        if !self.normalized_weights_cached {
            panic!("normalized weights are not cached");
        }

        &self.normalized_weights[player]
    }

    /// Returns the equity of each private hand of the given player.
    ///
    /// After mutating the current node, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    pub fn equity(&self, player: usize) -> Vec<f32> {
        if !self.normalized_weights_cached {
            panic!("normalized weights are not cached");
        }

        let num_private_hands = self.num_private_hands(player);
        let mut tmp = vec![0.0; num_private_hands];

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
            .enumerate()
            .map(|(i, &v)| {
                v as f32 * (self.weights[player][i] / self.normalized_weights[player][i]) + 0.5
            })
            .collect()
    }

    /// Returns the expected values of each private hand of the current player.
    ///
    /// Panics if the current node is a chance node.
    ///
    /// After mutating the current node, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    pub fn expected_values(&self) -> Vec<f32> {
        if self.is_chance_node() {
            panic!("chance node is not allowed");
        }

        if !self.is_solved {
            panic!("game is not solved");
        }

        if !self.normalized_weights_cached {
            panic!("normalized weights are not cached");
        }

        let num_actions = self.node().num_actions();
        let num_private_hands = self.num_private_hands(self.current_player());

        let expected_values_detail = self.expected_values_detail();
        let strategy = self.strategy();

        let mut ret = Vec::with_capacity(num_private_hands);
        for i in 0..num_private_hands {
            let mut expected_value = 0.0;
            for j in 0..num_actions {
                let index = i + j * num_private_hands;
                expected_value += expected_values_detail[index] * strategy[index];
            }
            ret.push(expected_value);
        }

        ret
    }

    /// Returns the expected values of each action of each private hand of the current player.
    ///
    /// Panics if the current node is a chance node.
    ///
    /// The return value is a vector of the length of `#(actions) * #(private hands)`.
    /// The expected value of `i`-th action with `j`-th private hand is stored in the
    /// `i * #(private hands) + j`-th element.
    ///
    /// After mutating the current node, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    pub fn expected_values_detail(&self) -> Vec<f32> {
        if self.is_chance_node() {
            panic!("chance node is not allowed");
        }

        if !self.is_solved {
            panic!("game is not solved");
        }

        if !self.normalized_weights_cached {
            panic!("normalized weights are not cached");
        }

        let player = self.current_player();

        let mut ret = if self.is_compression_enabled {
            let slice = self.node().expected_values_compressed();
            let scale = self.node().expected_value_scale();
            decode_signed_slice(slice, scale)
        } else {
            self.node().expected_values().to_vec()
        };

        let num_private_hands = self.num_private_hands(player);
        ret.chunks_mut(num_private_hands).for_each(|chunk| {
            self.apply_swap(chunk, player);
            chunk.iter_mut().enumerate().for_each(|(i, x)| {
                *x *= self.normalize_factor / self.normalized_weights[player][i];
                *x += self.config.starting_pot as f32 * 0.5 + self.node().amount as f32;
            });
        });

        ret
    }

    /// Returns the strategy of the current player.
    ///
    /// Panics if the current node is a chance node.
    ///
    /// The return value is a vector of the length of `#(actions) * #(private hands)`.
    /// The probability of `i`-th action with `j`-th private hand is stored in the
    /// `i * #(private hands) + j`-th element.
    pub fn strategy(&self) -> Vec<f32> {
        if self.is_chance_node() {
            panic!("chance node is not allowed");
        }

        if !self.is_solved {
            panic!("game is not solved");
        }

        let player = self.current_player();
        let num_actions = self.node().num_actions();
        let num_private_hands = self.num_private_hands(player);

        let mut ret = if self.is_compression_enabled {
            let slice = self.node().strategy_compressed();
            slice.iter().map(|&x| x as f32).collect()
        } else {
            self.node().strategy().to_vec()
        };

        normalize_strategy(&mut ret, num_actions);

        ret.chunks_mut(num_private_hands).for_each(|chunk| {
            self.apply_swap(chunk, player);
        });

        ret
    }

    /// Returns the reference to the current node.
    #[inline]
    fn node(&self) -> &PostFlopNode {
        unsafe { &*self.node }
    }

    /// Applies the swap.
    #[inline]
    fn apply_swap(&self, slice: &mut [f32], player: usize) {
        for swap in [self.river_swap, self.turn_swap] {
            if !swap.is_null() {
                for &(i, j) in unsafe { &(*swap)[player] } {
                    slice.swap(i, j);
                }
            }
        }
    }

    /// Internal method for calculating the equity.
    fn equity_internal(&self, result: &mut [f64], player: usize, turn: u8, river: u8, amount: f64) {
        let player_cards = &self.private_hand_cards[player];
        let opponent_cards = &self.private_hand_cards[player ^ 1];

        let opponent_weights = &self.weights[player ^ 1];
        let mut weight_sum = 0.0;
        let mut weight_minus = [0.0; 52];

        let hand_strength = &self.hand_strength[card_pair_index(turn, river)];
        let player_strength = &hand_strength[player];
        let opponent_strength = &hand_strength[player ^ 1];

        let mut j = 0;
        let player_len = player_strength.len();
        let opponent_len = opponent_strength.len();

        for i in 0..player_len {
            unsafe {
                let StrengthItem { strength, index } = *player_strength.get_unchecked(i);
                while j < opponent_len && opponent_strength.get_unchecked(j).strength < strength {
                    let opponent_index = opponent_strength.get_unchecked(j).index;
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let weight = *opponent_weights.get_unchecked(opponent_index) as f64;
                    weight_sum += weight;
                    *weight_minus.get_unchecked_mut(c1 as usize) += weight;
                    *weight_minus.get_unchecked_mut(c2 as usize) += weight;
                    j += 1;
                }
                let (c1, c2) = *player_cards.get_unchecked(index);
                let opponent_weight = weight_sum
                    - weight_minus.get_unchecked(c1 as usize)
                    - weight_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(index) += amount * opponent_weight;
            }
        }

        weight_sum = 0.0;
        weight_minus.fill(0.0);
        j = opponent_len;

        for i in (0..player_len).rev() {
            unsafe {
                let StrengthItem { strength, index } = *player_strength.get_unchecked(i);
                while j > 0 && opponent_strength.get_unchecked(j - 1).strength > strength {
                    let opponent_index = opponent_strength.get_unchecked(j - 1).index;
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let weight = *opponent_weights.get_unchecked(opponent_index) as f64;
                    weight_sum += weight;
                    *weight_minus.get_unchecked_mut(c1 as usize) += weight;
                    *weight_minus.get_unchecked_mut(c2 as usize) += weight;
                    j -= 1;
                }
                let (c1, c2) = *player_cards.get_unchecked(index);
                let opponent_weight = weight_sum
                    - weight_minus.get_unchecked(c1 as usize)
                    - weight_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(index) -= amount * opponent_weight;
            }
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
    fn strategy(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements) }
    }

    #[inline]
    fn cum_regret(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements) }
    }

    #[inline]
    fn cum_regret_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements) }
    }

    #[inline]
    fn expected_values(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements) }
    }

    #[inline]
    fn expected_values_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements) }
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u16, self.num_elements) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u16, self.num_elements) }
    }

    #[inline]
    fn cum_regret_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements) }
    }

    #[inline]
    fn cum_regret_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements) }
    }

    #[inline]
    fn expected_values_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements) }
    }

    #[inline]
    fn expected_values_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements) }
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
    fn cum_regret_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_cum_regret_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn expected_value_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_expected_value_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }
}

impl PostFlopNode {
    /// Returns the available actions.
    #[inline]
    fn available_actions(&self) -> Vec<Action> {
        self.children.iter().map(|(action, _)| *action).collect()
    }
}

impl Default for PostFlopGame {
    #[inline]
    fn default() -> Self {
        Self {
            config: GameConfig::default(),
            root: Box::default(),
            num_combinations_inv: 0.0,
            initial_weight: Default::default(),
            private_hand_cards: Default::default(),
            same_hand_index: Default::default(),
            hand_strength: Vec::default(),
            turn_isomorphism: Vec::default(),
            turn_isomorphism_cards: Vec::default(),
            turn_isomorphism_swap: Default::default(),
            river_isomorphism: Vec::default(),
            river_isomorphism_cards: Vec::default(),
            river_isomorphism_swap: Vec::default(),
            is_memory_allocated: false,
            is_compression_enabled: false,
            misc_memory_usage: 0,
            num_storage_elements: 0,
            storage1: MutexLike::default(),
            storage2: MutexLike::default(),
            storage1_compressed: MutexLike::default(),
            storage2_compressed: MutexLike::default(),
            is_solved: false,
            node: ptr::null(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
            weights: Default::default(),
            normalized_weights: Default::default(),
            normalized_weights_cached: false,
            normalize_factor: 0.0,
            turn_swapped_suit: None,
            turn_swap: ptr::null(),
            river_swap: ptr::null(),
        }
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
            storage1: ptr::null_mut(),
            storage2: ptr::null_mut(),
            scale1: 0.0,
            scale2: 0.0,
            num_elements: 0,
        }
    }
}

impl Default for GameConfig {
    #[inline]
    fn default() -> Self {
        Self {
            flop: [NOT_DEALT; 3],
            turn: NOT_DEALT,
            river: NOT_DEALT,
            starting_pot: 0,
            effective_stack: 0,
            range: Default::default(),
            flop_bet_sizes: Default::default(),
            turn_bet_sizes: Default::default(),
            river_bet_sizes: Default::default(),
            add_all_in_threshold: 0.0,
            force_all_in_threshold: 0.0,
            adjust_last_two_bet_sizes: false,
        }
    }
}

#[inline]
fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * mem::size_of::<T>() as u64
}

/// Attempts to convert an optionally space-separated string into a sorted flop array.
///
/// Card ID: `"2c"` => `0`, `"2d"` => `1`, `"2h"` => `2`, ..., `"As"` => `51`.
///
/// # Examples
/// ```
/// use postflop_solver::flop_from_str;
///
/// assert_eq!(flop_from_str("2c3d4h"), Ok([0, 5, 10]));
/// assert_eq!(flop_from_str("As Ah Ks"), Ok([47, 50, 51]));
/// ```
#[inline]
pub fn flop_from_str(s: &str) -> Result<[u8; 3], String> {
    let mut result = [0; 3];
    let mut chars = s.chars();

    result[0] = card_from_chars(&mut chars)?;
    result[1] = card_from_chars(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;
    result[2] = card_from_chars(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;

    if chars.next().is_some() {
        return Err("expected three cards".to_string());
    }

    result.sort_unstable();

    if result[0] == result[1] || result[1] == result[2] {
        return Err("cards must be unique".to_string());
    }

    Ok(result)
}

/// Attempts to convert a string into a card.
///
/// Card ID: `"2c"` => `0`, `"2d"` => `1`, `"2h"` => `2`, ..., `"As"` => `51`.
///
/// # Examples
/// ```
/// use postflop_solver::card_from_str;
///
/// assert_eq!(card_from_str("2c"), Ok(0));
/// assert_eq!(card_from_str("3d"), Ok(5));
/// assert_eq!(card_from_str("4h"), Ok(10));
/// assert_eq!(card_from_str("As"), Ok(51));
/// ```
#[inline]
pub fn card_from_str(s: &str) -> Result<u8, String> {
    let mut chars = s.chars();
    let result = card_from_chars(&mut chars)?;

    if chars.next().is_some() {
        return Err("expected two characters".to_string());
    }

    Ok(result)
}

#[inline]
fn card_from_chars<T: Iterator<Item = char>>(chars: &mut T) -> Result<u8, String> {
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
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        assert!((root_equity - 0.5).abs() < 1e-5);
        assert!((root_ev - 30.0).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: [Range::ones(); 2],
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        assert!((root_equity - 0.5).abs() < 1e-5);
        assert!((root_ev - 37.5).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range_compressed() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: [Range::ones(); 2],
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(true);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        assert!((root_equity - 0.5).abs() < 1e-4);
        assert!((root_ev - 37.5).abs() < 1e-2);
    }

    #[test]
    fn one_raise_all_range_with_turn() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            turn: card_from_str("Qc").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: [Range::ones(); 2],
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        assert!((root_equity - 0.5).abs() < 1e-5);
        assert!((root_ev - 37.5).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range_with_river() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            turn: card_from_str("Qc").unwrap(),
            river: card_from_str("7s").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: [Range::ones(); 2],
            river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        assert!((root_equity - 0.5).abs() < 1e-5);
        assert!((root_ev - 37.5).abs() < 1e-4);
    }

    #[test]
    fn always_win() {
        // be careful for straight flushes
        let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let config = GameConfig {
            flop: flop_from_str("AcAdKh").unwrap(),
            starting_pot: 60,
            effective_stack: 970,
            range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
            ..Default::default()
        };

        let mut game = PostFlopGame::with_config(&config).unwrap();
        game.allocate_memory(false);
        finalize(&mut game);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        assert!((root_equity - 1.0).abs() < 1e-5);
        assert!((root_ev - 60.0).abs() < 1e-4);
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
    fn solve_pio_preset() {
        let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
        let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

        let config = GameConfig {
            flop: flop_from_str("QsJh2h").unwrap(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
            starting_pot: 180,
            effective_stack: 910,
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
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

        let mut game = PostFlopGame::with_config(&config).unwrap();
        println!(
            "memory usage: {:.2}GB",
            game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        game.allocate_memory(false);

        solve(&mut game, 1000, 180.0 * 0.001, true);

        game.cache_normalized_weights();
        let weights = game.normalized_weights(game.current_player());

        let root_equity = compute_average(&game.equity(game.current_player()), weights);
        let root_ev = compute_average(&game.expected_values(), weights);

        // verified by PioSOLVER Free
        assert!((root_equity - 0.55347).abs() < 1e-5);
        assert!((root_ev - 105.11).abs() < 0.2);
    }
}
