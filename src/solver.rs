use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::ptr;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;
#[cfg(feature = "custom-alloc")]
use std::vec;

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    const ALPHA: f64 = 1.5;
    const GAMMA: f64 = 5.0;

    pub fn new(current_iteration: u32) -> Self {
        let float = (current_iteration as i32 - 1).max(0) as f64;

        let pow_alpha = float.powf(Self::ALPHA);
        let pow_gamma = (float / (float + 1.0)).powf(Self::GAMMA);

        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

/// Performs Discounted CFR algorithm until the given number of iterations or exploitability is
/// satisfied.
///
/// This method returns the exploitability of the obtained strategy.
pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if !game.is_ready() {
        panic!("the game is not ready");
    }

    if game.is_solved() {
        panic!("the game is already solved");
    }

    let mut root = game.root();
    let reach = [game.initial_weight(0), game.initial_weight(1)];

    let mut exploitability = compute_exploitability(game);

    if print_progress {
        print!("iteration: 0 / {} ", max_num_iterations);
        print!("(exploitability = {:.4e}[bb])", exploitability);
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        // alternating updates
        for player in 0..2 {
            let mut result = vec![0.0; game.num_private_hands(player)];
            solve_recursive(
                &mut result,
                game,
                &mut root,
                player,
                reach[player],
                reach[player ^ 1],
                &params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_num_iterations {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            print!("(exploitability = {:.4e}[bb])", exploitability);
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }

    finalize(game);

    exploitability
}

/// Proceeds Discounted CFR algorithm for one iteration.
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    if !game.is_ready() {
        panic!("the game is not ready");
    }

    if game.is_solved() {
        panic!("the game is already solved");
    }

    let mut root = game.root();
    let reach = [game.initial_weight(0), game.initial_weight(1)];
    let params = DiscountParams::new(current_iteration);

    // alternating updates
    for player in 0..2 {
        let mut result = vec![0.0; game.num_private_hands(player)];
        solve_recursive(
            &mut result,
            game,
            &mut root,
            player,
            reach[player],
            reach[player ^ 1],
            &params,
        );
    }
}

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [f32],
    game: &T,
    node: &mut T::Node,
    player: usize,
    reach: &[f32],
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // return the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_private_hands = game.num_private_hands(player);

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive(result, game, child, player, reach, cfreach, params);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(vec::from_elem_in(
        0.0,
        num_actions * num_private_hands,
        StackAlloc,
    ));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_private_hands]);

    // if the `node` is chance
    if node.is_chance() {
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = vec::from_elem_in(0.0, num_private_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = vec![0.0; num_private_hands];

        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                reach,
                &cfreach,
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            result_f64.iter_mut().zip(row).for_each(|(r, &v)| {
                *r += v as f64;
            });
        });

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for i in 0..isomorphic_chances.len() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(
                &mut cfv_actions,
                isomorphic_chances[i] as usize,
                num_private_hands,
            );

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
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

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            *r = v as f32;
        });
    }
    // if the current player is `player`
    else if node.player() == player {
        // compute the strategy by regret-maching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(
                node.cum_regret_compressed(),
                node.cum_regret_scale(),
                num_actions,
            )
        } else {
            regret_matching(node.cum_regret(), num_actions)
        };

        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut reach_actions = {
            let mut tmp = Vec::with_capacity_in(strategy.len(), StackAlloc);
            tmp.extend_from_slice(&strategy);
            tmp
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut reach_actions = strategy.clone();
        reach_actions.chunks_mut(num_private_hands).for_each(|row| {
            mul_slice(row, reach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                row(&reach_actions, action, num_private_hands),
                cfreach,
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        strategy
            .chunks(num_private_hands)
            .zip(cfv_actions.chunks(num_private_hands))
            .for_each(|(strategy_row, cfv_row)| {
                result
                    .iter_mut()
                    .zip(strategy_row)
                    .zip(cfv_row)
                    .for_each(|((r, &s), &v)| *r += s * v);
            });

        if game.is_compression_enabled() {
            // update the cumulative strategy
            let scale = node.strategy_scale();
            let cum_strategy = node.strategy_compressed_mut();
            let decoder = params.gamma_t * scale / u16::MAX as f32;

            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });

            let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
            node.set_strategy_scale(new_scale);

            // update the cumulative regret
            let scale = node.cum_regret_scale();
            let cum_regret = node.cum_regret_compressed_mut();
            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
            let beta_decoder = params.beta_t * scale / i16::MAX as f32;

            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                *x += if *y >= 0 {
                    (*y as f32) * alpha_decoder
                } else {
                    (*y as f32) * beta_decoder
                }
            });

            cfv_actions.chunks_mut(num_private_hands).for_each(|row| {
                sub_slice(row, result);
            });

            let new_scale = encode_signed_slice(cum_regret, &cfv_actions);
            node.set_cum_regret_scale(new_scale);
        } else {
            // update the cumulative strategy
            let cum_strategy = node.strategy_mut();
            mul_slice_scalar(cum_strategy, params.gamma_t);
            add_slice(cum_strategy, &strategy);

            // update the cumulative regret
            let cum_regret = node.cum_regret_mut();
            cum_regret.iter_mut().for_each(|el| {
                *el *= if *el >= 0.0 {
                    params.alpha_t
                } else {
                    params.beta_t
                };
            });
            add_slice(cum_regret, &cfv_actions);
            cum_regret.chunks_mut(num_private_hands).for_each(|row| {
                sub_slice(row, result);
            });
        }
    }
    // if the current player is not `player`
    else {
        // compute the strategy by regret-matching algorithm
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(
                node.cum_regret_compressed(),
                node.cum_regret_scale(),
                num_actions,
            )
        } else {
            regret_matching(node.cum_regret(), num_actions)
        };

        // update the reach probabilities
        let row_size = cfreach_actions.len() / num_actions;
        cfreach_actions.chunks_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                reach,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });

        // sum up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            add_slice(result, row);
        });
    }
}

/// Computes the strategy by regret-mathcing algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = regret.to_vec_in(StackAlloc);
    let row_size = strategy.len() / num_actions;

    strategy.iter_mut().for_each(|el| *el = el.max(0.0));

    let mut denom = vec::from_elem_in(0.0, row_size, StackAlloc);
    strategy.chunks(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-mathcing algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = regret.to_vec();
    let row_size = strategy.len() / num_actions;

    strategy.iter_mut().for_each(|el| *el = el.max(0.0));

    let mut denom = vec![0.0; row_size];
    strategy.chunks(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-mathcing algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching_compressed(
    regret: &[i16],
    scale: f32,
    num_actions: usize,
) -> Vec<f32, StackAlloc> {
    let mut strategy = decode_signed_slice_nonnegative(regret, scale);
    let row_size = strategy.len() / num_actions;

    let mut denom = vec::from_elem_in(0.0, row_size, StackAlloc);
    strategy.chunks(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-mathcing algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching_compressed(regret: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
    let mut strategy = decode_signed_slice_nonnegative(regret, scale);
    let row_size = strategy.len() / num_actions;

    let mut denom = vec![0.0; row_size];
    strategy.chunks(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Decodes the encoded `i16` slice to the non-negative `f32` slice.
#[cfg(feature = "custom-alloc")]
#[inline]
fn decode_signed_slice_nonnegative(slice: &[i16], scale: f32) -> Vec<f32, StackAlloc> {
    let decoder = scale / i16::MAX as f32;
    let mut result = Vec::<f32, StackAlloc>::with_capacity_in(slice.len(), StackAlloc);
    let ptr = result.as_mut_ptr();
    unsafe {
        for i in 0..slice.len() {
            *ptr.add(i) = (*slice.get_unchecked(i)).max(0) as f32 * decoder;
        }
        result.set_len(slice.len());
    }
    result
}

/// Decodes the encoded `i16` slice to the non-negative `f32` slice.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn decode_signed_slice_nonnegative(slice: &[i16], scale: f32) -> Vec<f32> {
    let decoder = scale / i16::MAX as f32;
    let mut result = Vec::<f32>::with_capacity(slice.len());
    let ptr = result.as_mut_ptr();
    unsafe {
        for i in 0..slice.len() {
            *ptr.add(i) = (*slice.get_unchecked(i)).max(0) as f32 * decoder;
        }
        result.set_len(slice.len());
    }
    result
}
