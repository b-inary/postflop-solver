use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::ptr;

#[cfg(feature = "custom-alloc")]
use {crate::alloc::*, std::vec};

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
    if game.is_solved() {
        panic!("the game is already solved");
    }

    if !game.is_ready() {
        panic!("the game is not ready");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);

    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        print!("(exploitability = {exploitability:.4e})");
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        // alternating updates
        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_num_iterations {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            print!("(exploitability = {exploitability:.4e})");
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
    if game.is_solved() {
        panic!("the game is already solved");
    }

    if !game.is_ready() {
        panic!("the game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);

    // alternating updates
    for player in 0..2 {
        let mut result = Vec::with_capacity(game.num_private_hands(player));
        solve_recursive(
            result.spare_capacity_mut(),
            game,
            &mut root,
            player,
            game.initial_weights(player ^ 1),
            &params,
        );
    }
}

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // return the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive(result, game, child, player, cfreach, params);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // if the `node` is chance
    if node.is_chance() {
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = vec::from_elem_in(0.0, num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = vec![0.0; num_hands];

        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach,
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        cfv_actions.chunks_exact(num_hands).for_each(|row| {
            result_f64.iter_mut().zip(row).for_each(|(r, &v)| {
                *r += v as f64;
            });
        });

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

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
            r.write(v as f32);
        });
    }
    // if the current player is `player`
    else if node.player() == player {
        // compute the strategy by regret-maching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), node.regret_scale(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        if game.is_compression_enabled() {
            // update the cumulative strategy
            let scale = node.strategy_scale();
            let decoder = params.gamma_t * scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();

            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });

            let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
            node.set_strategy_scale(new_scale);

            // update the cumulative regret
            let scale = node.regret_scale();
            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
            let beta_decoder = params.beta_t * scale / i16::MAX as f32;
            let cum_regret = node.regrets_compressed_mut();

            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                *x += *y as f32 * if *y >= 0 { alpha_decoder } else { beta_decoder };
            });

            cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });

            let new_scale = encode_signed_slice(cum_regret, &cfv_actions);
            node.set_regret_scale(new_scale);
        } else {
            // update the cumulative strategy
            let gamma_t = params.gamma_t;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x = *x * gamma_t + *y;
            });

            // update the cumulative regret
            let (alpha_t, beta_t) = (params.alpha_t, params.beta_t);
            let cum_regret = node.regrets_mut();
            cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                let coef = if x.is_sign_positive() {
                    alpha_t
                } else {
                    beta_t
                };
                *x = *x * coef + *y;
            });
            cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });
        }
    }
    // if the current player is not `player`
    else {
        // compute the strategy by regret-matching algorithm
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), node.regret_scale(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // update the reach probabilities
        let row_size = cfreach_actions.len() / num_actions;
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

#[inline]
fn max(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else {
        y
    }
}

/// Computes the strategy by regret-mathcing algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    let uninit = strategy.spare_capacity_mut();

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    denom.extend(regret[..row_size].iter().map(|el| max(*el, 0.0)));

    regret[row_size..].chunks_exact(row_size).for_each(|row| {
        add_slice_nonnegative(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(regret.chunks_exact(row_size))
        .for_each(|(s, r)| {
            div_slice_nonnegative_uninit(s, r, &denom, default);
        });

    unsafe { strategy.set_len(regret.len()) };
    strategy
}

/// Computes the strategy by regret-mathcing algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    let uninit = strategy.spare_capacity_mut();

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    denom.extend(regret[..row_size].iter().map(|el| max(*el, 0.0)));

    regret[row_size..].chunks_exact(row_size).for_each(|row| {
        add_slice_nonnegative(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(regret.chunks_exact(row_size))
        .for_each(|(s, r)| {
            div_slice_nonnegative_uninit(s, r, &denom, default);
        });

    unsafe { strategy.set_len(regret.len()) };
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

    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    denom.extend_from_slice(&strategy[..row_size]);
    strategy[row_size..].chunks_exact(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
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

    let mut denom = Vec::with_capacity(row_size);
    denom.extend_from_slice(&strategy[..row_size]);
    strategy[row_size..].chunks_exact(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
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
    let uninit = result.spare_capacity_mut();
    uninit.iter_mut().zip(slice).for_each(|(d, s)| {
        d.write((*s).max(0) as f32 * decoder);
    });
    unsafe { result.set_len(slice.len()) };
    result
}

/// Decodes the encoded `i16` slice to the non-negative `f32` slice.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn decode_signed_slice_nonnegative(slice: &[i16], scale: f32) -> Vec<f32> {
    let decoder = scale / i16::MAX as f32;
    let mut result = Vec::<f32>::with_capacity(slice.len());
    let uninit = result.spare_capacity_mut();
    uninit.iter_mut().zip(slice).for_each(|(d, s)| {
        d.write((*s).max(0) as f32 * decoder);
    });
    unsafe { result.set_len(slice.len()) }
    result
}
