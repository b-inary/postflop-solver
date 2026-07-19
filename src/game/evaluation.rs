use super::*;
use crate::card::NOT_DEALT;
use crate::hand::*;
use crate::sliceop::*;
use crate::utility::*;
use std::mem::MaybeUninit;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[inline]
fn min(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else {
        y
    }
}

impl PostFlopGame {
    pub(super) fn evaluate_internal(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
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
                &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
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
            let pair_index = card_pair_to_index(node.turn, node.river);
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

            let pair_index = card_pair_to_index(node.turn, node.river);
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

    pub(super) fn evaluate_internal_bunching(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = ((half_pot - rake) / self.bunching_num_combinations) as f32;
        let amount_lose = (-half_pot / self.bunching_num_combinations) as f32;
        let amount_tie = (-0.5 * rake / self.bunching_num_combinations) as f32;
        let opponent_len = self.private_cards[player ^ 1].len();

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let indices = if node.river != NOT_DEALT {
                &self.bunching_num_river[player][card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.bunching_num_turn[player][node.turn as usize]
            } else {
                &self.bunching_num_flop[player]
            };

            result.iter_mut().zip(indices).for_each(|(r, &index)| {
                if index != 0 {
                    let slice = &self.bunching_arena[index..index + opponent_len];
                    r.write(payoff * inner_product(cfreach, slice));
                } else {
                    r.write(0.0);
                }
            });
        }
        // showdown
        else {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let indices = &self.bunching_num_river[player][pair_index];
            let player_strength = &self.bunching_strength[pair_index][player];
            let opponent_strength = &self.bunching_strength[pair_index][player ^ 1];

            result
                .iter_mut()
                .zip(indices)
                .zip(player_strength)
                .for_each(|((r, &index), &strength)| {
                    if index != 0 {
                        r.write(inner_product_cond(
                            cfreach,
                            &self.bunching_arena[index..index + opponent_len],
                            opponent_strength,
                            strength,
                            amount_win,
                            amount_lose,
                            amount_tie,
                        ));
                    } else {
                        r.write(0.0);
                    }
                });
        }
    }

    /// 预计算全下权益矩阵。
    ///
    /// `equity_matrix[hi * num_villain + vi]` 存储 OOP 手牌 hi 对阵 IP 手牌 vi 的
    /// 全下期望收益（归一化），对所有可能的 turn+river 组合取平均。
    ///
    /// 对于 Turn DL，额外计算 `equity_matrix_turn[turn_card]`（52张转牌各自的矩阵）。
    pub(super) fn compute_equity_matrix(&mut self) {
        let num_oop = self.private_cards[0].len();
        let num_ip = self.private_cards[1].len();
        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        // 收集翻牌后剩余的牌
        let remaining: Vec<u8> = (0..52u8).filter(|c| (1u64 << c) & flop_mask == 0).collect();

        // 预计算翻牌基础手牌
        let flop_hand = Hand::new()
            .add_card(flop[0] as usize)
            .add_card(flop[1] as usize)
            .add_card(flop[2] as usize);

        // 翻牌级 matrix（1081 turn+river combos）
        let matrix = into_par_iter(0..num_oop)
            .map(|hi| {
                let (h1, h2) = self.private_cards[0][hi];
                let hero_mask = (1u64 << h1) | (1u64 << h2);
                let hero_base = flop_hand.add_card(h1 as usize).add_card(h2 as usize);

                let mut row = vec![0.0f32; num_ip];

                for vi in 0..num_ip {
                    let (v1, v2) = self.private_cards[1][vi];
                    let villain_mask = (1u64 << v1) | (1u64 << v2);

                    if hero_mask & villain_mask != 0 {
                        continue;
                    }

                    let known_mask = flop_mask | hero_mask | villain_mask;
                    let villain_base = flop_hand.add_card(v1 as usize).add_card(v2 as usize);

                    let mut wins = 0i32;
                    let mut losses = 0i32;
                    let mut count = 0u32;

                    for i in 0..remaining.len() {
                        let t = remaining[i];
                        if (1u64 << t) & known_mask != 0 {
                            continue;
                        }
                        let hero_5 = hero_base.add_card(t as usize);
                        let villain_5 = villain_base.add_card(t as usize);

                        for j in (i + 1)..remaining.len() {
                            let r = remaining[j];
                            if (1u64 << r) & known_mask != 0 {
                                continue;
                            }

                            let hero_rank = hero_5.add_card(r as usize).evaluate();
                            let villain_rank = villain_5.add_card(r as usize).evaluate();

                            if hero_rank > villain_rank {
                                wins += 1;
                            } else if hero_rank < villain_rank {
                                losses += 1;
                            }
                            count += 1;
                        }
                    }

                    if count > 0 {
                        row[vi] = (wins - losses) as f32 / count as f32;
                    }
                }

                row
            })
            .collect::<Vec<_>>();

        self.equity_matrix = matrix.into_iter().flatten().collect();

        // Turn DL: 为每张可能的转牌计算 turn-level equity matrix
        if self.tree_config.depth_limit == Some(BoardState::Turn) {
            self.equity_matrix_turn = vec![Vec::new(); 52];
            let num_oop = self.private_cards[0].len();
            let num_ip = self.private_cards[1].len();

            for turn_card in 0..52u8 {
                if (1u64 << turn_card) & flop_mask != 0 {
                    continue;
                }
                let turn_hand = flop_hand.add_card(turn_card as usize);

                let matrix_t = into_par_iter(0..num_oop)
                    .map(|hi| {
                        let (h1, h2) = self.private_cards[0][hi];
                        let hero_mask = (1u64 << h1) | (1u64 << h2);
                        if hero_mask & (1u64 << turn_card) != 0 {
                            return vec![0.0f32; num_ip];
                        }
                        let hero_6 = turn_hand.add_card(h1 as usize).add_card(h2 as usize);
                        let known = flop_mask | (1u64 << turn_card) | hero_mask;

                        let mut row = vec![0.0f32; num_ip];
                        for vi in 0..num_ip {
                            let (v1, v2) = self.private_cards[1][vi];
                            let vm = (1u64 << v1) | (1u64 << v2);
                            if vm & known == 0 {
                                let villain_6 = turn_hand.add_card(v1 as usize).add_card(v2 as usize);
                                let mut w = 0i32; let mut l = 0i32; let mut c = 0u32;
                                let all_known = known | vm;
                                for &r in &remaining {
                                    if (1u64 << r) & all_known != 0 { continue; }
                                    let hr = hero_6.add_card(r as usize).evaluate();
                                    let vr = villain_6.add_card(r as usize).evaluate();
                                    if hr > vr { w += 1 } else if hr < vr { l += 1 }
                                    c += 1;
                                }
                                if c > 0 { row[vi] = (w - l) as f32 / c as f32; }
                            }
                        }
                        row
                    }).collect::<Vec<_>>();
                self.equity_matrix_turn[turn_card as usize] = matrix_t.into_iter().flatten().collect();
            }
        }
    }

    /// 预计算手牌分类（成牌轴 + 听牌轴）。
    ///
    /// 在 `allocate_memory()` 中调用，分类结果存储于 `hand_categories` 和
    /// `hand_categories_turn`。求解过程中直接查表，无需重复计算。
    pub(super) fn compute_hand_categories(&mut self) {
        let flop = self.card_config.flop;
        let depth_limit = self.tree_config.depth_limit.unwrap();

        match depth_limit {
            BoardState::Flop => {
                // 翻牌截断：5 张已知牌（flop + 2 底牌）
                for player in 0..2 {
                    self.hand_categories[player] = self.private_cards[player]
                        .iter()
                        .map(|&cards| classify_hand(cards, flop, NOT_DEALT))
                        .collect();
                }
            }
            BoardState::Turn => {
                if self.card_config.turn != NOT_DEALT {
                    // 转牌已知：6 张已知牌
                    let turn = self.card_config.turn;
                    for player in 0..2 {
                        self.hand_categories[player] = self.private_cards[player]
                            .iter()
                            .map(|&cards| classify_hand(cards, flop, turn))
                            .collect();
                    }
                } else {
                    // 转牌随节点变化：预计算所有可能的 turn
                    let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
                    for player in 0..2 {
                        self.hand_categories_turn[player] = (0..52u8)
                            .map(|turn| {
                                if (1u64 << turn) & flop_mask != 0 {
                                    Vec::new()
                                } else {
                                    self.private_cards[player]
                                        .iter()
                                        .map(|&cards| classify_hand(cards, flop, turn))
                                        .collect()
                                }
                            })
                            .collect();
                    }
                }
            }
            BoardState::River => {}
        }
    }

    /// 获取当前 DL 节点对应的手牌分类表。
    #[inline]
    fn get_categories(&self, player: usize, node: &PostFlopNode) -> &[HandClassification] {
        match self.tree_config.depth_limit.unwrap() {
            BoardState::Flop => &self.hand_categories[player],
            BoardState::Turn => {
                if self.card_config.turn != NOT_DEALT {
                    &self.hand_categories[player]
                } else {
                    &self.hand_categories_turn[player][node.turn as usize]
                }
            }
            BoardState::River => unreachable!(),
        }
    }

    /// 深度限制终端节点求值（纯 equity 版本，不含位置和手牌系数）。
    ///
    /// `result[h] = Σ_v cfreach[v] × equity_matrix[h][v] × factor`
    pub(super) fn evaluate_depth_limited(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let factor = ((half_pot - 0.5 * rake) / self.num_combinations) as f32;

        let num_hero = self.private_cards[player].len();
        let num_villain = self.private_cards[player ^ 1].len();
        let hero_cards = &self.private_cards[player];
        let villain_cards = &self.private_cards[player ^ 1];

        // 初始化 result 为 0
        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });
        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        // 预处理：cfreach 的总和及按牌面扣减（用于 card blocking）
        let mut cfreach_sum = 0.0f64;
        let mut cfreach_minus = [0.0f64; 52];
        for vi in 0..num_villain {
            let cfreach_vi = cfreach[vi] as f64;
            if cfreach_vi != 0.0 {
                let (c1, c2) = villain_cards[vi];
                cfreach_sum += cfreach_vi;
                cfreach_minus[c1 as usize] += cfreach_vi;
                cfreach_minus[c2 as usize] += cfreach_vi;
            }
        }

        // Turn DL 时 turn 牌已是死牌，需要从 cfreach 扣除持有该牌的对手手牌
        let turn_blocker = if self.tree_config.depth_limit == Some(BoardState::Turn) && node.turn != NOT_DEALT {
            cfreach_minus[node.turn as usize]
        } else {
            0.0
        };

        for hi in 0..num_hero {
            let (c1, c2) = hero_cards[hi];
            let effective_cfreach_sum =
                cfreach_sum - cfreach_minus[c1 as usize] - cfreach_minus[c2 as usize] - turn_blocker;

            if effective_cfreach_sum == 0.0 {
                continue;
            }

            let mut cfv = 0.0f64;

            for vi in 0..num_villain {
                let (v1, v2) = villain_cards[vi];
                if (1u64 << v1 | 1u64 << v2) & (1u64 << c1 | 1u64 << c2) != 0 {
                    continue;
                }

                let cfreach_vi = cfreach[vi] as f64;
                if cfreach_vi == 0.0 {
                    continue;
                }

                // Turn DL 也使用 flop-level matrix：turn-level matrix 太精确导致值极端，
                // CFR 难以收敛。flop matrix 对所有 turn 取平均，更稳定。
                let matrix = &self.equity_matrix;
                let oop_equity = matrix[if player == 0 {
                    hi * num_villain + vi
                } else {
                    vi * num_hero + hi
                }];

                let base_equity = if player == 0 {
                    oop_equity as f64
                } else {
                    -(oop_equity as f64)
                };
                cfv += cfreach_vi * base_equity;
            }

            // 最终结果：cfv × factor
            let chip_result = cfv * factor as f64;

            // Fold Equity bonus: 按桶差异化,让CFR学会选择bluff手牌
            let abs_equity = if player == 0 {
                cfv.abs() / factor as f64
            } else {
                (-cfv).abs() / factor as f64
            };
            let opponent_air = self.air_ratio[player ^ 1];

            // 按桶确定 fold equity multiplier（桶6弱听牌最受益:有backdoor潜力）
            let (c1, c2) = hero_cards[hi];
            let flop = self.card_config.flop;
            let (cat, draw) = classify_hand((c1, c2), flop, self.card_config.turn);
            let bucket = HandBucket::classify(cat, draw, (c1, c2), flop);
            let bucket_factor: f64 = match bucket {
                HandBucket::WeakDraw => 0.22,   // 弱听牌: 最好的bluff候选
                HandBucket::Trash => 0.15,      // 纯空气: 次选bluff
                HandBucket::StrongDraw => 0.14, // 常规强听牌: 也可bluff
                HandBucket::MediumMade => 0.06, // 中等成牌: 少量bluff价值
                _ => 0.0,
            };

            let fe_bonus = if abs_equity < 0.6 && bucket_factor > 0.0 && player == 1 {
                (1.0 - abs_equity) * bucket_factor * opponent_air * (half_pot as f64)
            } else {
                0.0
            };

            result[hi] = (chip_result + fe_bonus) as f32;
        }
    }
}
