use crate::game::*;
use crate::interface::*;
use crate::sliceop::*;

type SwapList = [Vec<(usize, usize)>; 2];

/// A solved result interpreter of [`PostFlopGame`] struct.
pub struct Interpreter<'a> {
    game: &'a PostFlopGame,
    node: *const PostFlopNode,
    turn: u8,
    river: u8,
    weights: [Vec<f32>; 2],
    weights_normalized: [Vec<f64>; 2],
    weights_normalized_cached: bool,
    normalize_factor: f64,
    turn_swapped_suit: Option<(u8, u8)>,
    turn_swap: Option<&'a SwapList>,
    river_swap: Option<&'a SwapList>,
    ignore_threshold: f32,
}

unsafe impl<'a> Send for Interpreter<'a> {}
unsafe impl<'a> Sync for Interpreter<'a> {}

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

/// Decodes the encoded `u16` slice to the `f32` slice.
#[inline]
fn decode_unsigned_slice(slice: &[u16], scale: f32) -> Vec<f32> {
    let decoder = scale / u16::MAX as f32;
    let mut result = Vec::<f32>::with_capacity(slice.len());
    let ptr = result.as_mut_ptr();
    unsafe {
        for i in 0..slice.len() {
            *ptr.add(i) = *slice.get_unchecked(i) as f32 * decoder;
        }
        result.set_len(slice.len());
    }
    result
}

/// Computes the average with given weights.
#[inline]
pub fn compute_average<T: Copy + Into<f64>, U: Copy + Into<f64>>(
    slice: &[T],
    weights: &[U],
) -> f64 {
    let mut weight_sum = 0.0;
    let mut product_sum = 0.0;
    for (&v, &w) in slice.iter().zip(weights.iter()) {
        weight_sum += w.into();
        product_sum += v.into() * w.into();
    }
    product_sum / weight_sum
}

impl<'a> Interpreter<'a> {
    /// Creates a new interpreter for the given solved game.
    /// - `ignore_threshold`: the threshold of weight to ignore.
    ///   For example, if `ignore_threshold` is set to 0.001, the hands with weights less than 0.1%
    ///   are treated as weights of 0.
    #[inline]
    pub fn new(game: &'a PostFlopGame, ignore_threshold: f32) -> Self {
        if !game.is_solved() {
            panic!("game is not solved");
        }

        Self {
            game,
            node: &*game.root(),
            turn: game.config().turn,
            river: game.config().river,
            weights: [
                game.initial_weight(0).to_vec(),
                game.initial_weight(1).to_vec(),
            ],
            weights_normalized: [
                vec![0.0; game.num_private_hands(0)],
                vec![0.0; game.num_private_hands(1)],
            ],
            weights_normalized_cached: false,
            normalize_factor: game.num_combinations(),
            turn_swapped_suit: None,
            turn_swap: None,
            river_swap: None,
            ignore_threshold,
        }
    }

    /// Backs to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        self.node = &*self.game.root();
        self.turn = self.game.config().turn;
        self.river = self.game.config().river;
        self.weights = [
            self.game.initial_weight(0).to_vec(),
            self.game.initial_weight(1).to_vec(),
        ];
        self.weights_normalized_cached = false;
        self.normalize_factor = self.game.num_combinations();
        self.turn_swapped_suit = None;
        self.turn_swap = None;
        self.river_swap = None;
    }

    /// Returns the available actions.
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
    pub fn is_terminal(&self) -> Vec<bool> {
        self.node()
            .actions()
            .map(|action| {
                let child = self.node().play(action);
                child.is_terminal() || child.amount() == self.game.config().effective_stack
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
    ///
    /// Card ID: 2c2d2h2s => `0-3`, 3c3d3h3s => `4-7`, ..., AcAdAhAs => `48-51`.
    pub fn possible_cards(&self) -> u64 {
        if !self.node().is_chance() {
            return 0;
        }

        let flop = self.game.config().flop;
        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }

        let mut mask: u64 = (1 << 52) - 1;

        for (i, &(c1, c2)) in self.game.private_hand_cards(0).iter().enumerate() {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            let oop_weight = self.weights[0][i];
            if board_mask & oop_mask == 0 && oop_weight >= self.ignore_threshold {
                for (j, &(c3, c4)) in self.game.private_hand_cards(1).iter().enumerate() {
                    let ip_mask: u64 = (1 << c3) | (1 << c4);
                    let ip_weight = self.weights[1][j];
                    if (board_mask | oop_mask) & ip_mask == 0 && ip_weight >= self.ignore_threshold
                    {
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
                let isomorphism = self.game.isomorphic_chances(self.node());
                let isomorphic_cards = self.game.isomorphic_cards(self.node());
                for (i, &repr_index) in isomorphism.iter().enumerate() {
                    if action_card == isomorphic_cards[i] {
                        action_index = repr_index;
                        if is_turn {
                            self.turn_swap = Some(self.game.isomorphic_swap(self.node(), i));
                            if let Action::Chance(repr_card) = actions[repr_index] {
                                self.turn_swapped_suit = Some((action_card & 3, repr_card & 3));
                            }
                        } else {
                            self.river_swap = Some(self.game.isomorphic_swap(self.node(), i));
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
                if self.game.is_compression_enabled() {
                    let weights_raw = row(
                        self.node().strategy_compressed(),
                        action,
                        self.game.num_private_hands(player),
                    );
                    let scale = self.node().strategy_scale();
                    let mut weights = decode_unsigned_slice(weights_raw, scale);
                    self.apply_swap(&mut weights, player);
                    mul_slice(&mut self.weights[player], &weights);
                } else {
                    let mut weights = row(
                        self.node().strategy(),
                        action,
                        self.game.num_private_hands(player),
                    )
                    .to_vec();
                    self.apply_swap(&mut weights, player);
                    mul_slice(&mut self.weights[player], &weights);
                }
            }

            // updates the node
            self.node = &*self.node().play(action);
        }

        if self.node().is_terminal() || self.node().amount() == self.game.config().effective_stack {
            panic!("playing a terminal action is not allowed");
        }

        self.weights_normalized_cached = false;
    }

    /// Computes the normalized weights and caches them.
    ///
    /// After calling the `play()` method, this method must be called before calling
    /// `normalized_weights()`, `expected_values()`, or `equity()`.
    pub fn cache_normalized_weights(&mut self) {
        if self.weights_normalized_cached {
            return;
        }

        self.weights_normalized[0].fill(0.0);
        self.weights_normalized[1].fill(0.0);

        let private_hand_cards = [
            self.game.private_hand_cards(0),
            self.game.private_hand_cards(1),
        ];

        let mut board_mask: u64 = 0;
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }
        if self.river != NOT_DEALT {
            board_mask |= 1 << self.river;
        }

        for (i, &(c1, c2)) in private_hand_cards[0].iter().enumerate() {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            let oop_weight = self.weights[0][i];
            if board_mask & oop_mask == 0 && oop_weight > 0.0 {
                for (j, &(c3, c4)) in private_hand_cards[1].iter().enumerate() {
                    let ip_mask: u64 = (1 << c3) | (1 << c4);
                    let ip_weight = self.weights[1][j];
                    if (board_mask | oop_mask) & ip_mask == 0 && ip_weight > 0.0 {
                        let weight = oop_weight as f64 * ip_weight as f64;
                        self.weights_normalized[0][i] += weight;
                        self.weights_normalized[1][j] += weight;
                    }
                }
            }
        }

        self.weights_normalized_cached = true;
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
    /// After calling the `play()` method, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    #[inline]
    pub fn normalized_weights(&self, player: usize) -> &[f64] {
        if !self.weights_normalized_cached {
            panic!("normalized weights are not cached");
        }
        &self.weights_normalized[player]
    }

    /// Returns the expected values of each private hand of the current player.
    ///
    /// After calling the `play()` method, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    pub fn expected_values(&self) -> Vec<f32> {
        if !self.weights_normalized_cached {
            panic!("normalized weights are not cached");
        }

        let player = self.current_player();

        let mut ret = if self.game.is_compression_enabled() {
            let slice = self.node().expected_values_compressed();
            let scale = self.node().expected_value_scale();
            decode_signed_slice(slice, scale)
        } else {
            self.node().expected_values().to_vec()
        };

        self.apply_swap(&mut ret, player);

        ret.iter_mut().enumerate().for_each(|(i, x)| {
            *x *= (self.normalize_factor / self.weights_normalized[player][i]) as f32;
            *x += self.game.config().starting_pot as f32 / 2.0 + self.node().amount() as f32;
        });

        ret
    }

    /// Returns the equity of each private hand of the current player.
    ///
    /// After calling the `play()` method, you must call the `cache_normalized_weights()` method
    /// before calling this method.
    pub fn equity(&self) -> Vec<f32> {
        if !self.weights_normalized_cached {
            panic!("normalized weights are not cached");
        }

        let player = self.current_player();

        let mut ret = if self.game.is_compression_enabled() {
            let slice = self.node().equity_compressed();
            let scale = self.node().equity_scale();
            decode_signed_slice(slice, scale)
        } else {
            self.node().equity().to_vec()
        };

        self.apply_swap(&mut ret, player);

        ret.iter_mut().enumerate().for_each(|(i, x)| {
            *x *= (self.normalize_factor / self.weights_normalized[player][i]) as f32;
            *x += 0.5;
        });

        ret
    }

    /// Returns the strategy of the current player.
    ///
    /// The strategy is a vector of the length of `#(actions) * #(private hands)`.
    /// The probability of `i`-th action with `j`-th private hand is stored in the
    /// `i * #(private hands) + j`-th element.
    pub fn strategy(&mut self) -> Vec<f32> {
        let player = self.current_player();
        let num_actions = self.node().num_actions();
        let num_private_hands = self.game.num_private_hands(player);

        let mut ret = if num_actions == 1 {
            vec![1.0; num_private_hands]
        } else if self.game.is_compression_enabled() {
            let slice = self.node().strategy_compressed();
            let scale = self.node().strategy_scale();
            decode_unsigned_slice(slice, scale)
        } else {
            self.node().strategy().to_vec()
        };

        for i in 0..num_actions {
            self.apply_swap(row_mut(&mut ret, i, num_private_hands), player);
        }

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
        for swap in [self.river_swap, self.turn_swap].into_iter().flatten() {
            for &(i, j) in &swap[player] {
                slice.swap(i, j);
            }
        }
    }
}
