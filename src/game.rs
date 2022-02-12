use crate::bet_size::*;
use crate::interface::*;
use crate::mutex_like::*;
use crate::range::*;
use crate::utility::*;
use holdem_hand_evaluator::Hand;
use rayon::prelude::*;
use std::cmp::max;
use std::mem::{size_of, swap};
use std::slice::{from_raw_parts, from_raw_parts_mut};
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
    initial_reach: [Vec<f32>; 2],
    private_hand_cards: [Vec<(u8, u8)>; 2],
    same_hand_index: [Vec<Option<usize>>; 2],
    hand_strength: Vec<[Vec<(usize, usize)>; 2]>,

    // global storage
    cum_regret: MutexLike<Vec<f32>>,
    strategy: MutexLike<Vec<f32>>,
}

/// A struct representing a node in post-flop game tree.
pub struct PostFlopNode {
    player: u16,
    turn: u8,
    river: u8,
    amount: i32,
    children: Vec<(Action, MutexLike<PostFlopNode>)>,
    iso_chances: Vec<IsomorphicChance>,
    cum_regret: *mut f32,
    strategy: *mut f32,
    num_elements: usize,
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
///     initial_pot: 60,
///     initial_stack: 970,
///     range: [Range::ones(), Range::ones()],
///     flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     max_num_bet: 5,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GameConfig {
    /// Flop cards: each card must be unique and in range [0, 51].
    pub flop: [u8; 3],

    /// Initial pot size: must be greater than 0.
    pub initial_pot: i32,

    /// Initial effective stack size: must be greater than 0.
    pub initial_stack: i32,

    /// Initial range of each player.
    pub range: [Range; 2],

    /// Bet size candidates of each player in flop.
    pub flop_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player in turn.
    pub turn_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player in river.
    pub river_bet_sizes: [BetSizeCandidates; 2],

    /// Maximum number of bet in each betting round.
    pub max_num_bet: i32,
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
    AllIn,
    Chance(u8),
}

#[derive(Debug, Clone)]
struct BuildTreeInfo<'a> {
    last_action: Action,
    last_bet: [i32; 2],
    num_bet: i32,
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
fn align_up(size: usize) -> usize {
    size
}

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
    fn initial_reach(&self, player: usize) -> &[f32] {
        &self.initial_reach[player]
    }

    fn evaluate(&self, result: &mut [f32], node: &Self::Node, player: usize, cfreach: &[f32]) {
        let amount = self.config.initial_pot as f64 * 0.5 + node.amount as f64;
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
            let hand_strength = &self.hand_strength[board_index(node.turn, node.river)];
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
}

impl PostFlopGame {
    /// Constructs a new [`PostFlopGame`] instance with the given configuration.
    ///
    /// # Arguments
    /// - `config` - [`GameConfig`] instance.
    /// - `max_memory_mb` - Maximum amount of memory in megabytes.
    pub fn new(config: &GameConfig, max_memory_mb: Option<u32>) -> Result<Self, String> {
        let mut game = Self::default();
        game.update_config(config, max_memory_mb)?;
        Ok(game)
    }

    /// Updates the game configuration.
    pub fn update_config(
        &mut self,
        config: &GameConfig,
        max_memory_mb: Option<u32>,
    ) -> Result<(), String> {
        self.config = config.clone();
        self.check_config()?;
        self.init(max_memory_mb)?;
        Ok(())
    }

    /// Returns the card list of private hands of the given player.
    pub fn private_hand_cards(&self, player: usize) -> &Vec<(u8, u8)> {
        &self.private_hand_cards[player]
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

        if self.config.initial_pot <= 0 {
            return Err(format!(
                "Initial pot must be positive: initial_pot = {}",
                self.config.initial_pot
            ));
        }

        if self.config.initial_stack <= 0 {
            return Err(format!(
                "Initial stack must be positive: initial_stack = {}",
                self.config.initial_stack
            ));
        }

        if self.config.range[PLAYER_OOP as usize].is_empty() {
            return Err("OOP's range is not initialized".to_string());
        }

        if self.config.range[PLAYER_IP as usize].is_empty() {
            return Err("IP's range is not initialized".to_string());
        }

        let mut num_combinations = 0.0;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        for c1 in 0..52 {
            for c2 in c1 + 1..52 {
                let oop_mask: u64 = (1 << c1) | (1 << c2);
                let oop_prob = self.config.range[0].get_prob_by_cards(c1, c2);
                if oop_mask & flop_mask == 0 && oop_prob > 0.0 {
                    for c3 in 0..52 {
                        for c4 in c3 + 1..52 {
                            let ip_mask: u64 = (1 << c3) | (1 << c4);
                            let ip_prob = self.config.range[1].get_prob_by_cards(c3, c4);
                            if ip_mask & (flop_mask | oop_mask) == 0 {
                                num_combinations += oop_prob as f64 * ip_prob as f64;
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
    fn init(&mut self, max_memory_mb: Option<u32>) -> Result<(), String> {
        self.init_range();
        self.init_hand_strength();
        self.init_root(max_memory_mb)?;
        Ok(())
    }

    /// Initializes fields `initial_reach`, `private_hand_cards` and `same_hand_index`.
    fn init_range(&mut self) {
        let flop = self.config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        for player in 0..2 {
            let range = &self.config.range[player];
            let initial_reach = &mut self.initial_reach[player];
            let private_hand_cards = &mut self.private_hand_cards[player];
            initial_reach.clear();
            private_hand_cards.clear();

            for card1 in 0..52 {
                for card2 in card1 + 1..52 {
                    let hand_mask: u64 = (1 << card1) | (1 << card2);
                    let prob = range.get_prob_by_cards(card1, card2);
                    if prob > 0.0 && hand_mask & flop_mask == 0 {
                        initial_reach.push(prob);
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

    /// Initializes a field `hand_strength`.
    fn init_hand_strength(&mut self) {
        let mut flop = Hand::new();
        for card in &self.config.flop {
            flop = flop.add_card(*card as usize);
        }

        let private_hand_cards = &self.private_hand_cards;

        self.hand_strength = (0..52)
            .into_par_iter()
            .flat_map(|board1| {
                (board1 + 1..52).into_par_iter().map(move |board2| {
                    if !flop.contains(board1 as usize) && !flop.contains(board2 as usize) {
                        let board = flop.add_card(board1 as usize).add_card(board2 as usize);
                        let mut strength = [Vec::new(), Vec::new()];

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

                            strength[player].sort_unstable();
                        }

                        strength
                    } else {
                        Default::default()
                    }
                })
            })
            .collect();
    }

    /// Initializes the root node of game tree.
    fn init_root(&mut self, max_memory_mb: Option<u32>) -> Result<(), String> {
        let current_memory_usage = AtomicU64::new(size_of::<PostFlopNode>() as u64);
        let num_storage_elements = AtomicU64::new(0);
        let max_stack_size = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let max_num_private_hands = max(self.num_private_hands(0), self.num_private_hands(1));

        let info = BuildTreeInfo {
            last_action: Action::None,
            last_bet: [0, 0],
            num_bet: 0,
            allin_flag: false,
            current_memory_usage: &current_memory_usage,
            num_storage_elements: &num_storage_elements,
            stack_size: [size_of::<f32>() * max_num_private_hands; 2],
            max_stack_size: &max_stack_size,
        };

        self.root().children.clear();
        self.build_tree_recursive(&mut self.root(), &info);

        let stack_size = max(
            max_stack_size[0].load(Ordering::Relaxed),
            max_stack_size[1].load(Ordering::Relaxed),
        );

        #[cfg(feature = "custom_alloc")]
        // heuristically the stack size is multiplied by 8
        // (there is no guarantee that the stack size is enough with rayon library)
        STACK_SIZE.store(8 * stack_size, Ordering::Relaxed);

        let current_memory_usage = current_memory_usage.load(Ordering::Relaxed);
        let num_storage_elements = num_storage_elements.load(Ordering::Relaxed);
        let storage_size = 2 * size_of::<f32>() as u64 * num_storage_elements;
        let stack_coef = if cfg!(feature = "custom_alloc") { 8 } else { 3 };
        let stack_usage = (stack_coef * stack_size * rayon::current_num_threads()) as u64;
        let total_memory_usage = current_memory_usage + storage_size + stack_usage;

        let memory_limit = max_memory_mb
            .map(|mb| (mb as u64) << 20)
            .unwrap_or(usize::MAX as u64);

        // memory usage check
        if total_memory_usage > memory_limit {
            return Err(format!(
                "Memory usage {:.2}GB exceeds the limit {:.2}GB",
                total_memory_usage as f64 / (1 << 30) as f64,
                memory_limit as f64 / (1 << 30) as f64
            ));
        }

        self.cum_regret.lock().clear();
        self.cum_regret.lock().shrink_to_fit();
        self.cum_regret = MutexLike::new(vec![0.0; num_storage_elements as usize]);

        self.strategy.lock().clear();
        self.strategy.lock().shrink_to_fit();
        self.strategy = MutexLike::new(vec![0.0; num_storage_elements as usize]);

        let counter = AtomicUsize::new(0);
        self.allocate_memory_recursive(&mut self.root(), &counter);

        Ok(())
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
            let f32_size = size_of::<f32>();
            let col_size = f32_size * node.num_actions();
            for i in 0..2 {
                stack_size[i] += align_up(col_size * self.num_private_hands(i));
                stack_size[i] += align_up(f32_size * self.num_private_hands(i ^ 1));
            }

            for_each_child(node, |index| {
                let (last_action, child) = &node.children[index];
                self.build_tree_recursive(
                    &mut child.lock(),
                    &BuildTreeInfo {
                        last_action: *last_action,
                        last_bet: [0, 0],
                        stack_size,
                        ..*info
                    },
                );
            });
        }
        // player node
        else {
            self.push_actions(node, info);

            let mut stack_size = info.stack_size;
            let col_size = size_of::<f32>() * node.num_actions();
            for i in 0..2 {
                let n = if i == node.player() { 2 } else { 1 };
                stack_size[i] += align_up(col_size * self.num_private_hands(i));
                stack_size[i] += n * align_up(col_size * self.num_private_hands(node.player()));
            }

            for_each_child(node, |index| {
                let (action, child) = &node.children[index];
                let mut last_bet = info.last_bet;
                let mut num_bet = info.num_bet;
                let mut allin_flag = info.allin_flag;

                match *action {
                    Action::Call => {
                        last_bet[node.player as usize] = last_bet[node.player as usize ^ 1];
                    }
                    Action::Bet(size) | Action::Raise(size) => {
                        last_bet[node.player as usize] = size;
                        num_bet += 1;
                    }
                    Action::AllIn => {
                        last_bet[node.player as usize] += self.config.initial_stack - node.amount;
                        num_bet += 1;
                        allin_flag = true;
                    }
                    _ => {}
                }

                self.build_tree_recursive(
                    &mut child.lock(),
                    &BuildTreeInfo {
                        last_action: *action,
                        last_bet,
                        num_bet,
                        allin_flag,
                        stack_size,
                        ..*info
                    },
                )
            });
        }
    }

    /// Pushes the chance actions to the `node`.
    fn push_chances(&self, node: &mut PostFlopNode, info: &BuildTreeInfo) {
        let flop = self.config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let mut indices = [0; 52];

        let mut flop_rankset = [0; 4];
        for card in flop {
            let rank = card >> 2;
            let suit = card & 3;
            flop_rankset[suit as usize] |= 1 << rank;
        }

        // deal turn
        if node.turn == NOT_DEALT {
            let next_player = if !info.allin_flag {
                PLAYER_OOP
            } else {
                PLAYER_CHANCE
            };

            let mut iso_suits = [None; 4];
            for suit1 in 0..4 {
                for suit2 in suit1 + 1..4 {
                    if iso_suits[suit2 as usize].is_none()
                        && flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                    {
                        iso_suits[suit2 as usize] = Some(suit1);
                    }
                }
            }

            for card in 0..52 {
                if (1 << card) & flop_mask != 0 {
                    continue;
                }

                let rank = card >> 2;
                let suit = card & 3;

                // isomorphic chance
                if let Some(iso_suit) = iso_suits[suit as usize] {
                    let iso_card = rank << 2 | iso_suit;
                    let iso_index = indices[iso_card as usize];
                    let mut iso_chance = IsomorphicChance {
                        index: iso_index,
                        swap_list: Default::default(),
                    };

                    for player in 0..2 {
                        let cards = &self.private_hand_cards[player];
                        for i in 0..cards.len() {
                            let (c1, c2) = cards[i];
                            if c1 == card {
                                if let Ok(j) = cards.binary_search(&(iso_card, c2)) {
                                    iso_chance.swap_list[player].push((i, j));
                                }
                            }
                            if c2 == card {
                                if let Ok(j) = cards.binary_search(&(c1, iso_card)) {
                                    iso_chance.swap_list[player].push((i, j));
                                }
                            }
                        }
                        iso_chance.swap_list[player].shrink_to_fit();
                        info.current_memory_usage.fetch_add(
                            vec_memory_usage(&iso_chance.swap_list[player]),
                            Ordering::Relaxed,
                        );
                    }

                    node.iso_chances.push(iso_chance);
                }
                // normal chance
                else {
                    indices[card as usize] = node.children.len();
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
            let mut turn_rankset = flop_rankset;
            turn_rankset[node.turn as usize & 3] |= 1 << (node.turn >> 2);

            let next_player = if !info.allin_flag {
                PLAYER_OOP
            } else {
                PLAYER_TERMINAL_FLAG
            };

            let mut iso_suits = [None; 4];
            for suit1 in 0..4 {
                for suit2 in suit1 + 1..4 {
                    if iso_suits[suit2 as usize].is_none()
                        && flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                        && turn_rankset[suit1 as usize] == turn_rankset[suit2 as usize]
                    {
                        iso_suits[suit2 as usize] = Some(suit1);
                    }
                }
            }

            for card in 0..52 {
                if (1 << card) & turn_mask != 0 {
                    continue;
                }

                let rank = card >> 2;
                let suit = card & 3;

                // isomorphic chance
                if let Some(iso_suit) = iso_suits[suit as usize] {
                    let iso_card = rank << 2 | iso_suit;
                    let iso_index = indices[iso_card as usize];
                    let mut iso_chance = IsomorphicChance {
                        index: iso_index,
                        swap_list: Default::default(),
                    };

                    for player in 0..2 {
                        let cards = &self.private_hand_cards[player];
                        for i in 0..cards.len() {
                            let (c1, c2) = cards[i];
                            if c1 == card {
                                if let Ok(j) = cards.binary_search(&(iso_card, c2)) {
                                    iso_chance.swap_list[player].push((i, j));
                                }
                            }
                            if c2 == card {
                                if let Ok(j) = cards.binary_search(&(c1, iso_card)) {
                                    iso_chance.swap_list[player].push((i, j));
                                }
                            }
                        }
                        iso_chance.swap_list[player].shrink_to_fit();
                        info.current_memory_usage.fetch_add(
                            vec_memory_usage(&iso_chance.swap_list[player]),
                            Ordering::Relaxed,
                        );
                    }

                    node.iso_chances.push(iso_chance);
                }
                // normal chance
                else {
                    indices[card as usize] = node.children.len();
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
        node.iso_chances.shrink_to_fit();

        info.current_memory_usage.fetch_add(
            vec_memory_usage(&node.children) + vec_memory_usage(&node.iso_chances),
            Ordering::Relaxed,
        );
    }

    /// Pushes the actions to the `node`.
    fn push_actions(&self, node: &mut PostFlopNode, info: &BuildTreeInfo) {
        let player = node.player;
        let player_opponent = node.player ^ 1;

        let player_bet = info.last_bet[player as usize];
        let opponent_bet = info.last_bet[player_opponent as usize];

        let bet_diff = opponent_bet - player_bet;
        let pot = self.config.initial_pot + 2 * (node.amount + bet_diff);

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
            if info.num_bet < self.config.max_num_bet {
                for &bet_size in &candidates[player as usize].bet {
                    match bet_size {
                        BetSize::PotRelative(ratio) => {
                            let size = (pot as f32 * ratio).round() as i32;
                            actions.push((Action::Bet(size), player_opponent));
                        }
                        BetSize::LastBetRelative(_) => panic!("unexpected bet size"),
                    }
                }
            }
        } else {
            // add fold
            actions.push((Action::Fold, PLAYER_FOLD_FLAG | player));

            // add call
            actions.push((Action::Call, player_after_call));

            // add raise
            if !info.allin_flag && info.num_bet < self.config.max_num_bet {
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
            }
        }

        let max_bet = self.config.initial_stack - node.amount + player_bet;
        let min_bet = max_bet.min(opponent_bet + bet_diff);

        // adjust bet sizes
        for (action, _) in actions.iter_mut() {
            match *action {
                Action::Bet(size) => {
                    let adjusted_size = size.clamp(min_bet, max_bet);
                    if adjusted_size == max_bet {
                        *action = Action::AllIn;
                    } else if size != adjusted_size {
                        *action = Action::Bet(adjusted_size);
                    }
                }
                Action::Raise(size) => {
                    let adjusted_size = size.clamp(min_bet, max_bet);
                    if adjusted_size == max_bet {
                        *action = Action::AllIn;
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
                Action::Call | Action::Bet(_) | Action::Raise(_) | Action::AllIn
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
            let cum_regret_ptr = self.cum_regret.lock().as_mut_ptr();
            let strategy_ptr = self.strategy.lock().as_mut_ptr();
            let index = counter.fetch_add(node.num_elements, Ordering::SeqCst);
            unsafe {
                node.cum_regret = cum_regret_ptr.add(index);
                node.strategy = strategy_ptr.add(index);
            }
        }

        for_each_child(node, |action| {
            self.allocate_memory_recursive(&mut node.play(action), counter);
        });
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
    fn isomorphic_chances(&self) -> &Vec<IsomorphicChance> {
        &self.iso_chances
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children[action].1.lock()
    }

    #[inline]
    fn cum_regret(&self) -> &[f32] {
        unsafe { from_raw_parts(self.cum_regret, self.num_elements) }
    }

    #[inline]
    fn cum_regret_mut(&mut self) -> &mut [f32] {
        unsafe { from_raw_parts_mut(self.cum_regret, self.num_elements) }
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        unsafe { from_raw_parts(self.strategy, self.num_elements) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        unsafe { from_raw_parts_mut(self.strategy, self.num_elements) }
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }
}

impl Default for PostFlopNode {
    fn default() -> Self {
        Self {
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            amount: 0,
            children: Vec::new(),
            iso_chances: Vec::new(),
            cum_regret: std::ptr::null_mut(),
            strategy: std::ptr::null_mut(),
            num_elements: 0,
        }
    }
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            flop: [NOT_DEALT; 3],
            initial_pot: 0,
            initial_stack: 0,
            range: Default::default(),
            flop_bet_sizes: Default::default(),
            turn_bet_sizes: Default::default(),
            river_bet_sizes: Default::default(),
            max_num_bet: 0,
        }
    }
}

#[inline]
fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * size_of::<T>() as u64
}

#[inline]
fn board_index(mut turn: u8, mut river: u8) -> usize {
    if turn > river {
        swap(&mut turn, &mut river);
    }
    turn as usize * (101 - turn as usize) / 2 + river as usize - 1
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
            initial_pot: 60,
            initial_stack: 970,
            range: [Range::ones(); 2],
            ..Default::default()
        };
        let game = PostFlopGame::new(&config, None).unwrap();
        normalize_strategy(&game);
        let ev0 = compute_ev(&game, 0) + 30.0;
        let ev1 = compute_ev(&game, 1) + 30.0;
        assert!((ev0 - 30.0).abs() < 1e-4);
        assert!((ev1 - 30.0).abs() < 1e-4);
    }

    #[test]
    fn one_raise_all_range() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            initial_pot: 60,
            initial_stack: 970,
            range: [Range::ones(); 2],
            river_bet_sizes: [bet_sizes_from_str("50%", "").unwrap(), Default::default()],
            max_num_bet: 1,
            ..Default::default()
        };
        let game = PostFlopGame::new(&config, None).unwrap();
        normalize_strategy(&game);
        let ev0 = compute_ev(&game, 0) + 30.0;
        let ev1 = compute_ev(&game, 1) + 30.0;
        assert!((ev0 - 37.5).abs() < 1e-4);
        assert!((ev1 - 22.5).abs() < 1e-4);
    }

    #[test]
    fn always_win() {
        // be careful for straight flushes
        let lose_range_str = "22+,A2+,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
        let config = GameConfig {
            flop: flop_from_str("AcAdKh").unwrap(),
            initial_pot: 60,
            initial_stack: 970,
            range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
            ..Default::default()
        };
        let game = PostFlopGame::new(&config, None).unwrap();
        normalize_strategy(&game);
        let ev0 = compute_ev(&game, 0) + 30.0;
        let ev1 = compute_ev(&game, 1) + 30.0;
        assert!((ev0 - 60.0).abs() < 1e-4);
        assert!((ev1 - 0.0).abs() < 1e-4);
    }

    #[test]
    fn no_assignment() {
        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            initial_pot: 60,
            initial_stack: 970,
            range: ["TT".parse().unwrap(), "TT".parse().unwrap()],
            ..Default::default()
        };
        let game = PostFlopGame::new(&config, None);
        assert!(game.is_err());
    }

    #[test]
    #[ignore]
    fn cfr_solve() {
        // top-40% range
        let oop_range =
            "22+,A2s+,A8o+,K7s+,K9o+,Q8s+,Q9o+,J8s+,J9o+,T8+,97+,86+,75+,64s+,65o,54,43s";

        // top-25% range
        let ip_range = "22+,A4s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9+,T9,98s,87s,76s,65s";

        let bet_sizes = bet_sizes_from_str("50%", "50%").unwrap();

        let config = GameConfig {
            flop: flop_from_str("Td9d6h").unwrap(),
            initial_pot: 60,
            initial_stack: 770,
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            max_num_bet: 5,
        };

        let game = PostFlopGame::new(&config, Some(3072)).unwrap();
        solve(&game, 1000, 60.0 * 0.005, true);
        let ev0 = compute_ev(&game, 0) + 30.0;
        let ev1 = compute_ev(&game, 1) + 30.0;

        // verified by GTO+
        assert!((ev0 - 26.24).abs() < 0.5);
        assert!((ev1 - 33.76).abs() < 0.5);
    }
}
