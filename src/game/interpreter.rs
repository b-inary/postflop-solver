use super::*;
use crate::interface::*;
use crate::sliceop::*;
use crate::utility::*;

/// Decodes the encoded `i16` slice to the `f32` slice.
#[inline]
fn decode_signed_slice(slice: &[i16], scale: f32) -> Vec<f32> {
    let decoder = scale / i16::MAX as f32;
    slice.iter().map(|&x| x as f32 * decoder).collect()
}

impl PostFlopGame {
    /// Moves the current node back to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.action_history.clear();
        self.node_history.clear();
        self.is_normalized_weight_cached = false;
        self.turn = self.card_config.turn;
        self.river = self.card_config.river;
        self.turn_swapped_suit = None;
        self.turn_swap = None;
        self.river_swap = None;
        self.total_bet_amount = [0, 0];

        self.weights[0].copy_from_slice(&self.initial_weights[0]);
        self.weights[1].copy_from_slice(&self.initial_weights[1]);
        self.assign_zero_weights();
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

        &self.action_history
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

    /// If the current node is a chance node, returns a list of cards that can be dealt.
    ///
    /// The returned value is a 64-bit integer.
    /// The `i`-th bit is set to 1 if the card of ID `i` can be dealt (see [`Card`] for encoding).
    /// If the current node is not a chance node, `0` is returned.
    pub fn possible_cards(&self) -> u64 {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if !self.is_chance_node() {
            return 0;
        }

        let flop = self.card_config.flop;
        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        let mut dead_mask: u64 = 0;

        // no bunching
        if self.bunching_num_dead_cards == 0 {
            if self.turn != NOT_DEALT {
                board_mask |= 1 << self.turn;
            }

            'outer: for card in 0..52 {
                let bit_card: u64 = 1 << card;
                let new_board_mask = board_mask | bit_card;

                if new_board_mask != board_mask {
                    for &(c1, c2) in &self.private_cards[0] {
                        let oop_mask: u64 = (1 << c1) | (1 << c2);
                        if oop_mask & new_board_mask != 0 {
                            continue;
                        }
                        let combined_mask = oop_mask | new_board_mask;
                        for &(c3, c4) in &self.private_cards[1] {
                            let ip_mask: u64 = (1 << c3) | (1 << c4);
                            if ip_mask & combined_mask == 0 {
                                continue 'outer;
                            }
                        }
                    }
                }

                dead_mask |= bit_card;
            }
        }
        // bunching
        else {
            let node_turn = self.node().turn;
            if node_turn != NOT_DEALT {
                board_mask |= 1 << node_turn;
            }

            let ip_len = self.num_private_hands(1);
            let mut children = Vec::new();
            let (iso_ref, iso_card) = if node_turn == NOT_DEALT {
                (&self.isomorphism_ref_turn, &self.isomorphism_card_turn)
            } else {
                (
                    &self.isomorphism_ref_river[node_turn as usize],
                    &self.isomorphism_card_river[node_turn as usize & 3],
                )
            };

            'outer: for card in 0..52 {
                let bit_card: u64 = 1 << card;
                let new_board_mask = board_mask | bit_card;

                if let Some(pos) = iso_card.iter().position(|&c| c == card) {
                    let ref_card = children[iso_ref[pos] as usize];
                    dead_mask |= ((dead_mask >> ref_card) & 1) << card;
                    continue;
                }

                if new_board_mask != board_mask {
                    children.push(card);
                    let indices = if node_turn == NOT_DEALT {
                        &self.bunching_num_turn[0][card as usize]
                    } else {
                        &self.bunching_num_river[0][card_pair_to_index(node_turn, card)]
                    };
                    for &index in indices {
                        if index == 0 {
                            continue;
                        }
                        let slice = &self.bunching_arena[index..index + ip_len];
                        if slice.iter().any(|&n| n > 0.0) {
                            continue 'outer;
                        }
                    }
                }

                dead_mask |= bit_card;
            }

            if let Some((suit1, suit2)) = self.turn_swapped_suit {
                let suit_mask: u64 = 0x1_1111_1111_1111;
                let mod_mask = (suit_mask << suit1) | (suit_mask << suit2);
                let swapped1 = ((dead_mask >> suit1) & suit_mask) << suit2;
                let swapped2 = ((dead_mask >> suit2) & suit_mask) << suit1;
                dead_mask = (dead_mask & !mod_mask) | swapped1 | swapped2;
            }
        }

        ((1 << 52) - 1) ^ dead_mask
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
    ///   - If the current node is a chance node, the `action` corresponds to the dealt card (see
    ///     [`Card`] for encoding). If `usize::MAX` is passed, the card is selected as the possible
    ///     card with the lowest index.
    ///   - If the current node is not a chance node, plays the `action`-th action of
    ///     [`available_actions`].
    ///
    /// Panics if the memory is not yet allocated or the current node is a terminal node.
    ///
    /// **Time complexity:** *O*(#(OOP private hands) + #(IP private hands))
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
            if self.storage_mode == BoardState::Flop
                || (!is_turn && self.storage_mode == BoardState::Turn)
            {
                panic!("Storage mode is not compatible");
            }

            let actual_card = if action == usize::MAX {
                self.possible_cards().trailing_zeros() as Card
            } else {
                action as Card
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
                let node = self.node();
                let isomorphism = self.isomorphic_chances(&node);
                let isomorphic_cards = if node.turn == NOT_DEALT {
                    &self.isomorphism_card_turn
                } else {
                    &self.isomorphism_card_river[node.turn as usize & 3]
                };
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
                                self.turn & 3,
                                self.isomorphism_card_river[self.turn as usize & 3][i] & 3,
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

            // update the state
            let node_index = self.node_index(&self.node().play(action_index));
            self.node_history.push(node_index);
            if is_turn {
                self.turn = actual_card;
            } else {
                self.river = actual_card;
            }

            // update the weights
            self.assign_zero_weights();
        }
        // player node
        else {
            // panic if the action is invalid
            let node = self.node();
            if action >= node.num_actions() {
                panic!("Invalid action");
            }

            let player = node.player();
            let num_hands = self.num_private_hands(player);

            // update the weights
            if node.num_actions() > 1 {
                let strategy = self.strategy();
                let weights = row(&strategy, action, num_hands);
                mul_slice(&mut self.weights[player], weights);
            }

            // cache the counterfactual values
            let node = self.node();
            let vec = if self.is_compression_enabled {
                let slice = row(node.cfvalues_compressed(), action, num_hands);
                let scale = node.cfvalue_scale();
                decode_signed_slice(slice, scale)
            } else {
                row(node.cfvalues(), action, num_hands).to_vec()
            };
            self.cfvalues_cache[player].copy_from_slice(&vec);

            // update the bet amounts
            let node = self.node();
            match node.play(action).prev_action {
                Action::Call => {
                    self.total_bet_amount[player] = self.total_bet_amount[player ^ 1];
                }
                Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => {
                    let prev_bet_amount = match node.prev_action {
                        Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => a,
                        _ => 0,
                    };
                    let to_call = self.total_bet_amount[player ^ 1] - self.total_bet_amount[player];
                    self.total_bet_amount[player] += amount - prev_bet_amount + to_call;
                }
                _ => {}
            }

            // update the node
            let node_index = self.node_index(&self.node().play(action));
            self.node_history.push(node_index);
        }

        self.action_history.push(action);
        self.is_normalized_weight_cached = false;
    }

    /// Computes the normalized weights and caches them.
    ///
    /// After mutating the current node, this method must be called once before calling
    /// [`normalized_weights`], [`equity`], [`expected_values`], or [`expected_values_detail`].
    ///
    /// **Time complexity:**
    /// - (no bunching) *O*(#(OOP private hands) + #(IP private hands))
    /// - (bunching) *O*(#(OOP private hands) * #(IP private hands))
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

        // no bunching
        if self.bunching_num_dead_cards == 0 {
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
        }
        // bunching
        else {
            let mut weights_buf = [Vec::new(), Vec::new()];
            let weights = if self.turn_swap.is_none() && self.river_swap.is_none() {
                &self.weights
            } else {
                weights_buf[0].extend_from_slice(&self.weights[0]);
                weights_buf[1].extend_from_slice(&self.weights[1]);
                self.apply_swap(&mut weights_buf[0], 0, true);
                self.apply_swap(&mut weights_buf[1], 1, true);
                &weights_buf
            };

            for player in 0..2 {
                let node = self.node();
                let indices = if node.river != NOT_DEALT {
                    &self.bunching_num_river[player][card_pair_to_index(node.turn, node.river)]
                } else if node.turn != NOT_DEALT {
                    &self.bunching_num_turn[player][node.turn as usize]
                } else {
                    &self.bunching_num_flop[player]
                };

                let opponent_len = self.num_private_hands(player ^ 1);
                let mut normalized_weights = indices
                    .iter()
                    .zip(weights[player].iter())
                    .map(|(&index, &w)| {
                        if index != 0 {
                            let slice = &self.bunching_arena[index..index + opponent_len];
                            w * inner_product(&weights[player ^ 1], slice)
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();

                self.apply_swap(&mut normalized_weights, player, false);
                self.normalized_weights[player] = normalized_weights;
            }
        }

        self.is_normalized_weight_cached = true;
    }

    /// Returns the weights of each private hand of the given player.
    ///
    /// If a hand overlaps with the board, returns 0.0.
    ///
    /// **Time complexity:** *O*(1).
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
    /// **Time complexity:** *O*(1).
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
    /// **Time complexity:**
    /// - (no bunching) *O*(#(possible 5-card boards) * (#(OOP private hands) + #(IP private hands))).
    /// - (bunching) *O*(#(OOP private hands) * #(IP private hands)).
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

        let tmp = if self.bunching_num_dead_cards == 0 {
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
            tmp.into_iter().map(|v| v as f32).collect()
        } else {
            let mut tmp = self.equity_internal_bunching(player);
            self.apply_swap(&mut tmp, player, false);
            tmp
        };

        tmp.iter()
            .zip(self.weights[player].iter())
            .zip(self.normalized_weights[player].iter())
            .map(|((&v, &w_raw), &w_normalized)| {
                if w_normalized > 0.0 {
                    v * (w_raw / w_normalized) + 0.5
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
    /// **Time complexity:** see [`expected_values_detail`].
    ///
    /// [`cache_normalized_weights`]: #method.cache_normalized_weights
    /// [`expected_values_detail`]: #method.expected_values_detail
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
    /// **Time complexity:**
    /// - (with bunching and the current node is terminal) *O*(#(OOP private hands) * #(IP private hands)).
    /// - (otherwise) *O*(#(actions) * #(private hands)).
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

        let node = self.node();
        let num_hands = self.num_private_hands(player);

        let mut chance_factor = 1;
        if self.card_config.turn == NOT_DEALT && self.turn != NOT_DEALT {
            chance_factor *= 45 - self.bunching_num_dead_cards;
        }
        if self.card_config.river == NOT_DEALT && self.river != NOT_DEALT {
            chance_factor *= 44 - self.bunching_num_dead_cards;
        }

        let num_combinations = match self.bunching_num_dead_cards {
            0 => self.num_combinations,
            _ => self.bunching_num_combinations,
        };

        let mut have_actions = false;
        let mut normalizer = (num_combinations * chance_factor as f64) as f32;

        let mut ret = if node.is_terminal() {
            normalizer = num_combinations as f32;
            let mut ret = Vec::with_capacity(num_hands);
            let mut cfreach = self.weights[player ^ 1].clone();
            self.apply_swap(&mut cfreach, player ^ 1, true);
            self.evaluate(ret.spare_capacity_mut(), &node, player, &cfreach);
            unsafe { ret.set_len(num_hands) };
            ret
        } else if node.is_chance() && node.cfvalue_storage_player() == Some(player) {
            if self.is_compression_enabled {
                let slice = node.cfvalues_chance_compressed();
                let scale = node.cfvalue_chance_scale();
                decode_signed_slice(slice, scale)
            } else {
                node.cfvalues_chance().to_vec()
            }
        } else if node.has_cfvalues_ip() && player == PLAYER_IP as usize {
            if self.is_compression_enabled {
                let slice = node.cfvalues_ip_compressed();
                let scale = node.cfvalue_ip_scale();
                decode_signed_slice(slice, scale)
            } else {
                node.cfvalues_ip().to_vec()
            }
        } else if player == self.current_player() {
            have_actions = true;
            if self.is_compression_enabled {
                let slice = node.cfvalues_compressed();
                let scale = node.cfvalue_scale();
                decode_signed_slice(slice, scale)
            } else {
                node.cfvalues().to_vec()
            }
        } else {
            self.cfvalues_cache[player].to_vec()
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
    ///
    /// **Time complexity:** *O*(#(actions) * #(private hands)).
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

        let node = self.node();
        let player = self.current_player();
        let num_actions = node.num_actions();
        let num_hands = self.num_private_hands(player);

        let mut ret = if self.is_compression_enabled {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        let locking = self.locking_strategy(&node);
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
    ///
    /// This method must be called after allocating memory and before solving the game.
    /// Panics if the memory is not yet allocated or the game is already solved.
    /// Also, panics if the current node is a terminal node or a chance node.
    pub fn lock_current_strategy(&mut self, strategy: &[f32]) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.state == State::Solved {
            panic!("Game is already solved");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let mut node = self.node();
        let player = self.current_player();
        let num_actions = node.num_actions();
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

        node.is_locked = true;
        let index = self.node_index(&node);
        self.locking_strategy.insert(index, locking);
    }

    /// Unlocks the strategy of the current node.
    ///
    /// This method must be called after allocating memory and before solving the game.
    /// Panics if the memory is not yet allocated or the game is already solved.
    /// Also, panics if the current node is a terminal node or a chance node.
    #[inline]
    pub fn unlock_current_strategy(&mut self) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.state == State::Solved {
            panic!("Game is already solved");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let mut node = self.node();
        if !node.is_locked {
            return;
        }

        node.is_locked = false;
        let index = self.node_index(&node);
        self.locking_strategy.remove(&index);
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

        let index = self.node_index(&self.node());
        self.locking_strategy.get(&index).map(|s| {
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
    fn node(&self) -> MutexGuardLike<PostFlopNode> {
        self.node_arena[self.node_history.last().cloned().unwrap_or(0)].lock()
    }

    /// Returns the index of the given node.
    #[inline]
    pub(super) fn node_index(&self, node: &PostFlopNode) -> usize {
        let node_ptr = node as *const _ as *const MutexLike<PostFlopNode>;
        unsafe { node_ptr.offset_from(self.node_arena.as_ptr()) as usize }
    }

    /// Assigns zero weights to the hands that are not possible.
    pub(super) fn assign_zero_weights(&mut self) {
        if self.bunching_num_dead_cards == 0 {
            let mut board_mask: u64 = 0;
            if self.turn != NOT_DEALT {
                board_mask |= 1 << self.turn;
            }
            if self.river != NOT_DEALT {
                board_mask |= 1 << self.river;
            }

            for player in 0..2 {
                let mut dead_mask: u64 = (1 << 52) - 1;

                for &(c1, c2) in &self.private_cards[player ^ 1] {
                    let mask: u64 = (1 << c1) | (1 << c2);
                    if mask & board_mask == 0 {
                        dead_mask &= mask;
                    }
                    if dead_mask == 0 {
                        break;
                    }
                }

                dead_mask |= board_mask;

                self.private_cards[player]
                    .iter()
                    .zip(self.weights[player].iter_mut())
                    .for_each(|(&(c1, c2), w)| {
                        let mask: u64 = (1 << c1) | (1 << c2);
                        if mask & dead_mask != 0 {
                            *w = 0.0;
                        }
                    });
            }
        } else {
            for player in 0..2 {
                let node = self.node();
                let opponent_len = self.num_private_hands(player ^ 1);
                let indices = if node.turn == NOT_DEALT {
                    &self.bunching_num_flop[player]
                } else if node.river == NOT_DEALT {
                    &self.bunching_num_turn[player][node.turn as usize]
                } else {
                    &self.bunching_num_river[player][card_pair_to_index(node.turn, node.river)]
                };

                let mut weights_buf = Vec::new();
                let weights = if self.turn_swap.is_none() && self.river_swap.is_none() {
                    &mut self.weights[player]
                } else {
                    weights_buf.extend_from_slice(&self.weights[player]);
                    self.apply_swap(&mut weights_buf, player, true);
                    &mut weights_buf
                };

                for (w, &index) in weights.iter_mut().zip(indices.iter()) {
                    if index == 0 {
                        *w = 0.0;
                    } else {
                        let slice = &self.bunching_arena[index..index + opponent_len];
                        if slice.iter().all(|&n| n == 0.0) {
                            *w = 0.0;
                        }
                    }
                }

                if self.turn_swap.is_some() || self.river_swap.is_some() {
                    self.apply_swap(&mut weights_buf, player, false);
                    self.weights[player].copy_from_slice(&weights_buf);
                }
            }
        }
    }

    /// Applies the swap.
    #[inline]
    fn apply_swap(&self, slice: &mut [f32], player: usize, reverse: bool) {
        let turn_swap = self
            .turn_swap
            .map(|suit| &self.isomorphism_swap_turn[suit as usize][player]);

        let river_swap = self.river_swap.map(|(turn_suit, suit)| {
            &self.isomorphism_swap_river[turn_suit as usize][suit as usize][player]
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
    fn equity_internal(
        &self,
        result: &mut [f64],
        player: usize,
        turn: Card,
        river: Card,
        amount: f64,
    ) {
        let pair_index = card_pair_to_index(turn, river);
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

    /// Internal method for calculating the equity.
    fn equity_internal_bunching(&self, player: usize) -> Vec<f32> {
        let mut weights_buf = Vec::new();
        let opponent_weights = if self.turn_swap.is_none() && self.river_swap.is_none() {
            &self.weights[player ^ 1]
        } else {
            weights_buf.extend_from_slice(&self.weights[player ^ 1]);
            self.apply_swap(&mut weights_buf, player ^ 1, true);
            &weights_buf
        };

        let node = self.node();
        let opponent_len = opponent_weights.len();

        if node.river == NOT_DEALT {
            let indices = if node.turn != NOT_DEALT {
                &self.bunching_coef_turn[player][node.turn as usize]
            } else {
                &self.bunching_coef_flop[player]
            };

            indices
                .iter()
                .map(|&index| {
                    if index != 0 {
                        let slice = &self.bunching_arena[index..index + opponent_len];
                        0.5 * inner_product(opponent_weights, slice)
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        // showdown
        else {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let indices = &self.bunching_num_river[player][pair_index];
            let player_strength = &self.bunching_strength[pair_index][player];
            let opponent_strength = &self.bunching_strength[pair_index][player ^ 1];

            indices
                .iter()
                .zip(player_strength)
                .map(|(&index, &strength)| {
                    if index != 0 {
                        inner_product_cond(
                            opponent_weights,
                            &self.bunching_arena[index..index + opponent_len],
                            opponent_strength,
                            strength,
                            0.5,
                            -0.5,
                            0.0,
                        )
                    } else {
                        0.0
                    }
                })
                .collect()
        }
    }
}
